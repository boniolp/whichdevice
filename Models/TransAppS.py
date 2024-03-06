import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Layers.Encoding import LearnablePositionalEncoding1D, PositionalEncoding1D, tAPE
from Models.Layers.Transformer import EncoderLayer
from Models.Layers.ConvLayer import DilatedBlock, InceptionModule

    
# ======================= TransAppV1 =======================#
class TransAppS(nn.Module):
    def __init__(self, 
                 mode="classif",
                 window_size=128,
                 c_in=1,  nb_class=2, 
                 instance_norm=False,
                 kernel_size=3, kernel_size_head=3, 
                 encoder_encoding_type='tAPE',
                 decoder_encoding_type='tAPE',
                 n_encoder_layers=1, n_decoder_layers=0,
                 d_model=96, dp_rate=0.2, activation='gelu',
                 pffn_ratio=4, n_head=8, prenorm=True, norm="LayerNorm", store_att=False, attn_dp_rate=0.2, 
                 att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False,
                            'attdec_mask_diag': True,  'attdec_mask_flag': False, 'learnable_scale_dec': False},
                 masking_type='random', mask_ratio=0.5, c_reconstruct=1, mask_mean_length=5, loss_in_model=nn.MSELoss()):
        super().__init__()
  
        self.mode          = mode
        self.d_model       = d_model
        self.instance_norm = instance_norm

        self.c_in          = c_in
        self.c_reconstruct = c_reconstruct
        
        self.mask_ratio       = mask_ratio
        self.mask_mean_length = mask_mean_length
        self.masking_type     = masking_type
        self.loss_in_model    = loss_in_model
        
        
        #============ Embedding ============#
        self.EmbedBlock = InceptionModule(in_channels=c_in, n_filters=d_model//4, bottleneck_channels=d_model//4)           
            
        #============ Positional Encoding Encoder ============#
        if encoder_encoding_type == 'learnable':
            self.PosEncoding_Encoder = LearnablePositionalEncoding1D(d_model, max_len=window_size)
        elif encoder_encoding_type == 'fixed':
            self.PosEncoding_Encoder = PositionalEncoding1D(d_model)
        elif encoder_encoding_type == 'tAPE':
            self.PosEncoding_Encoder = tAPE(d_model, max_len=window_size)
        elif encoder_encoding_type == 'noencoding':
            self.PosEncoding_Encoder = None
        else:
            raise ValueError('Type of encoding {} unknown, only "learnable", "fixed" or "noencoding" supported.'
                             .format(encoder_encoding_type))

        #============ Positional Encoding Decoder ============#
        if decoder_encoding_type == 'learnable':
            self.PosEncoding_Decoder = LearnablePositionalEncoding1D(d_model, max_len=window_size)
        elif decoder_encoding_type == 'fixed':
            self.PosEncoding_Decoder = PositionalEncoding1D(d_model)
        elif decoder_encoding_type == 'tAPE':
            self.PosEncoding_Decoder = tAPE(d_model, max_len=window_size)
        elif decoder_encoding_type == 'noencoding':
            self.PosEncoding_Decoder = None
        else:
            raise ValueError('Type of encoding {} unknown, only "learnable", "fixed" or "noencoding" supported.'
                             .format(decoder_encoding_type))
        
        #============ Encoder ============#
        layers = []
        for _ in range(n_encoder_layers):
            layers.append(EncoderLayer(d_model, d_model * pffn_ratio, n_head, 
                                       dp_rate=dp_rate, attn_dp_rate=attn_dp_rate, 
                                       att_mask_diag=att_param['attenc_mask_diag'], 
                                       att_mask_flag=att_param['attenc_mask_flag'], 
                                       learnable_scale=att_param['learnable_scale_enc'], 
                                       store_att=store_att,  norm=norm, prenorm=prenorm, activation=activation))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = torch.nn.Sequential(*layers)
        
        #============ Decoder ============#
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        layers = []
        for _ in range(n_decoder_layers):
            layers.append(EncoderLayer(d_model, d_model * pffn_ratio, n_head, 
                                       dp_rate=dp_rate, attn_dp_rate=attn_dp_rate,
                                       att_mask_diag=att_param['attdec_mask_diag'], 
                                       att_mask_flag=att_param['attdec_mask_flag'], 
                                       learnable_scale=att_param['learnable_scale_dec'], 
                                       store_att=store_att,  norm=norm, prenorm=prenorm, activation=activation))
        layers.append(nn.LayerNorm(d_model))
        self.PretrainingDecoder = torch.nn.Sequential(*layers)

        self.PredHead = nn.Conv1d(in_channels=d_model, out_channels=c_reconstruct, kernel_size=kernel_size_head, padding=kernel_size_head//2, padding_mode='replicate')
        
        #============ Classif Head ============#
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.LinearHead = nn.Linear(d_model, nb_class, bias=True)
        self.Dropout_Head = (nn.Dropout(dp_rate))

        #============ Stats Proj ============#
        self.ProjStats = nn.Linear(2, d_model)

        #============ Initializing weights ============#        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def freeze_params(self, model_part, rq_grad=False):
        for _, child in model_part.named_children():
            for param in child.parameters():
                param.requires_grad = rq_grad
            self.freeze_params(child)
            
    def get_mask(self, device, N, L):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        return mask, ids_keep and ids_restore
        """
        if self.masking_type=='subseq':
            # Masking using subsequences and Poisson Law
            mask = np.ones(L, dtype=bool)
            p_m = 1 / self.mask_mean_length
            p_u = p_m * self.mask_ratio / (1 - self.mask_ratio) 
            p = [p_m, p_u]
            
            state = int(np.random.rand() > self.mask_ratio)
            for i in range(L):
                mask[i] = state
                if np.random.rand() < p[state]:
                    state = 1 - state

            mask = torch.Tensor(mask).int().to(device)
            
            ids_keep    = torch.nonzero(mask, as_tuple=True)[0]
            ids_removed = torch.nonzero(~mask.bool(), as_tuple=True)[0]
            ids_shuffle = torch.cat((ids_keep, ids_removed))
            ids_restore = torch.argsort(ids_shuffle).unsqueeze(0).repeat(N, 1)

            mask = ((~mask.bool()).int()).unsqueeze(0).repeat(N, 1)
            ids_keep = ids_keep.unsqueeze(0).repeat(N, 1)

        else:
            len_keep = int(L * (1 - self.mask_ratio))
            
            # Get Noise in [0, 1] of shape [B, L]
            noise = torch.rand(N, L, device=device)  # noise in [0, 1]
            
            # Sort noise for each sample using acsending: small is keep, large is remove
            ids_shuffle = torch.argsort(noise, dim=1)  
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]

            # Generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=device)
            mask[:, :len_keep] = 0

            # Unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_keep, ids_restore

    def mask_data(self, x, ids_keep):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        _, _, D = x.shape  # Get B L D
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # Keep only selected tokens

        return x_masked

    def forward_encoder(self, x, ids_keep=None) -> torch.Tensor:
        # Remove masked tokens (timestamp)
        if self.mode=="pretraining" or self.mode=="output_mask":
            x = self.mask_data(x, ids_keep)

        # Forward Transformer Block
        x = self.EncoderBlock(x)
            
        return x
    
    def forward_decoder_pretraining(self, x, ids_restore) -> torch.Tensor:
        # Add Mask Token
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        #x = torch.cat([x, encoding.permute(0, 2, 1)], dim=2)

        # Transformer Block
        x = self.PretrainingDecoder(x)
            
        return x

    def forward(self, x) -> torch.Tensor:
        # Input as B 1+e L 
        # Separate load curve and embedding
        if x.shape[1]>1:
            encoding = x[:, 1:, :] # B 1 L
            x        = x[:, :1, :] # B e L
            sep = True
        else:
            sep = False

        if self.loss_in_model is not None:
            x_input = x.clone()

        # === Instance Normalization === #
        if self.instance_norm:
            inst_mean = torch.mean(x, dim=-1, keepdim=True).detach() # Mean: B 1 1
            inst_std  = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6).detach() # STD: B 1 1

            x = (x - inst_mean) / inst_std # Instance z-norm: B 1 1

        if sep:
            x = torch.cat((x, encoding), dim=1)
        
        # Forward Decoder
        if self.mode=="pretraining" or self.mode=="output_mask":
            # === Masked Auto Encoder === # 
            # Get mask for MAE training
            mask, ids_keep, ids_restore = self.get_mask(x.device, x.shape[0], x.shape[-1]) # B L
            # Set to 0 masked value to avoid leak of information using convolution
            x = x * (~(mask.bool())).int().unsqueeze(1) # B D L

            # === Embedding === # 
            # Conv dilated embedding block
            x = self.EmbedBlock(x).permute(0, 2, 1) # B L D
            # Add positional encoding
            if self.PosEncoding_Encoder is not None:
                x = self.PosEncoding_Encoder(x) # B D L

            # === Forward Transformer Encoder === #
            x = self.forward_encoder(x, ids_keep) # B L D

            # === Forward Transformer Decoder === #
            if self.PosEncoding_Encoder is not None:
                x = self.PosEncoding_Encoder(x) # B L D
            x = self.forward_decoder_pretraining(x, ids_restore) # B L D

            # === Conv Head === #
            x = x.permute(0, 2, 1) # B D L
            x = self.PredHead(x) # B 1 L

            # === Reverse Instance Normalization === #
            if self.instance_norm:
                x = x * inst_std + inst_mean

            if self.training:
                if self.loss_in_model is not None:
                    x_input = x_input * mask.unsqueeze(1)
                    x       = x * mask.unsqueeze(1)
                    loss    = self.loss_in_model(x, x_input)

                    return x, loss
                else:
                    return x
            else:
                if self.mode=="output_mask":
                    return x, mask
                else:
                    return x
                    
        else:
            # === Embedding === # 
            # Conv Dilated Embedding Block for aggregate
            x = self.EmbedBlock(x).permute(0, 2, 1) # B L D
            # Add positional encoding
            if self.PosEncoding_Encoder is not None:
                x = self.PosEncoding_Encoder(x) # B L D

            # === Mean and Std projection === #
            if self.instance_norm:
                stats_token = self.ProjStats(torch.cat([inst_mean, inst_std], dim=1).permute(0, 2, 1)) # B 1 D
                x = torch.cat([x, stats_token], dim=1) # Add stats token B L+1 D

            # === Forward Transformer Encoder === #
            x = self.forward_encoder(x) 
            if self.instance_norm:
                x = x[:, :-1, :] # Remove stats token: B L D

            # === GAP Head === #
            x = x.permute(0, 2, 1) # B D L
            x = self.GAP(x).flatten(start_dim=1) # B D -> B nb_class (using GAP)
            x = self.LinearHead(x)

            return x