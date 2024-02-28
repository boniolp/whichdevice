import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Layers.UtilsLayer import Conv1dSamePadding, pass_through


# ================ ResUnit Layer ================ #
class ResUnit(nn.Module):       
    def __init__(self, c_in, c_out, k=8, dilation=1, stride=1, bias=True):
        super().__init__()
        
        self.layers = nn.Sequential(Conv1dSamePadding(in_channels=c_in, out_channels=c_out,
                                                      kernel_size=k, dilation=dilation, stride=stride, bias=bias),
                                    nn.GELU(),
                                    nn.BatchNorm1d(c_out)
                                    )
        if c_in > 1 and c_in!=c_out:
            self.match_residual=True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual=False
            
    def forward(self,x):
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)
            
            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))


# ================ Dilated block using exponential dilation parameter ================ #
class DilatedBlock(nn.Module):  
    def __init__(self, c_in=32, c_out=32, 
                 kernel_size=8, dilation_list=[1, 2, 4, 8]):
        super().__init__()
 
        layers = []
        for i, dilation in enumerate(dilation_list):
            if i==0:
                layers.append(ResUnit(c_in, c_out, k=kernel_size, dilation=dilation))
            else:
                layers.append(ResUnit(c_out, c_out, k=kernel_size, dilation=dilation))
        self.network = torch.nn.Sequential(*layers)
            
    def forward(self,x):
        x = self.network(x)
        return x
    

# ================ Inception Module ================ #
class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], 
                 bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
        """
        : param in_channels          Number of input channels (input features)
        : param n_filters            Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes         List of kernel sizes for each convolution.
                                     Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                     This is nessesery because of padding size.
                                     For correction of kernel_sizes use function "correct_sizes". 
        : param bottleneck_channels  Number of output channels in bottleneck. 
                                     Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation           Activation function for output tensor (nn.ReLU()). 
        : param return_indices       Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
        """
        super(InceptionModule, self).__init__()
        self.return_indices=return_indices
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                                in_channels=in_channels, 
                                out_channels=bottleneck_channels, 
                                kernel_size=1, 
                                stride=1, 
                                bias=False
                                )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[0], 
                                        stride=1, 
                                        padding=kernel_sizes[0]//2, 
                                        bias=False
                                        )
        self.conv_from_bottleneck_2 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[1], 
                                        stride=1, 
                                        padding=kernel_sizes[1]//2, 
                                        bias=False
                                        )
        self.conv_from_bottleneck_3 = nn.Conv1d(
                                        in_channels=bottleneck_channels, 
                                        out_channels=n_filters, 
                                        kernel_size=kernel_sizes[2], 
                                        stride=1, 
                                        padding=kernel_sizes[2]//2, 
                                        bias=False
                                        )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=n_filters, 
                                    kernel_size=1, 
                                    stride=1,
                                    padding=0, 
                                    bias=False
                                    )
        self.batch_norm = nn.BatchNorm1d(num_features=(len(kernel_sizes)+1)*n_filters)
        self.activation = activation

    def forward(self, X):
        # step 1
        Z_bottleneck = self.bottleneck(X)
        if self.return_indices:
            Z_maxpool, indices = self.max_pool(X)
        else:
            Z_maxpool = self.max_pool(X)
        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        # step 3 
        Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        Z = self.activation(self.batch_norm(Z))
        if self.return_indices:
            return Z, indices
        else:
            return Z