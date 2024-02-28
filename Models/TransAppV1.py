import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Layers.Encoding import LearnablePositionalEncoding1D, PositionalEncoding1D, tAPE
from Models.Layers.Transformer import EncoderLayer
from Models.Layers.ConvLayer import DilatedBlock, InceptionModule
from Models.Layers.UtilsLayer import Transpose  

    
# ======================= TransAppV1 =======================#
class TransAppV1(nn.Module):
    def __init__(self, 
                 mode="classif",
                 window_size=128, 
                 c_in=1, c_out=1, instance_norm=True,
                 n_embed_blocks=1, type_embedding_block='conv', 
                 kernel_size=5, nb_class=2,
                 encoder_encoding_type='tAPE',
                 decoder_encoding_type='tAPE',
                 n_encoder_layers=2, n_decoder_layers=1,
                 d_model=64, dp_rate=0.2, activation='gelu',
                 pffn_ratio=4, n_head=8, prenorm=True, norm="LayerNorm", store_att=False, attn_dp_rate=0.2, 
                 att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False,
                            'attdec_mask_diag': True,  'attdec_mask_flag': False, 'learnable_scale_dec': False},
                 masking_type='random', mask_ratio=0.5, c_reconstruct=1, mask_mean_length=5, loss_in_model=False):
        super().__init__()
  
        self.c_in = c_in
        self.c_out = c_out
        self.c_reconstruct = c_reconstruct
        self.d_model = d_model
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.mask_mean_length = mask_mean_length
        self.masking_type = masking_type
        self.loss_in_model = loss_in_model
        self.instance_norm = instance_norm

        if kernel_size==1 and type_embedding_block!='conv':
            type_embedding_block = 'conv'
            warnings.warn("kernel_size=1 incompatible with type_embedding_block={}, type_embedding_block set to 'conv'.".format(type_embedding_block))

        self.type_embedding_block = type_embedding_block
            
        #============ Embedding ============#
        layers = []
        for i in range(n_embed_blocks):
            if type_embedding_block=='dilated':
                layers.append(DilatedBlock(c_in=c_in if i==0 else d_model, c_out=d_model, kernel_size=kernel_size, dilation_list=[1, 2, 4, 8]))
            elif type_embedding_block=='conv':
                layers.append(nn.Conv1d(in_channels=c_in if i==0 else d_model, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='replicate'))
            elif type_embedding_block=='inception':
                layers.append(InceptionModule(in_channels=c_in if i==0 else d_model, n_filters=d_model//4, bottleneck_channels=d_model//4))
        layers.append(Transpose(1, 2))
        self.EmbedBlock = torch.nn.Sequential(*layers) 
            
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
        for i in range(n_encoder_layers):
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
        for i in range(n_decoder_layers):
            layers.append(EncoderLayer(d_model, d_model * pffn_ratio, n_head, 
                                       dp_rate=dp_rate, attn_dp_rate=attn_dp_rate,
                                       att_mask_diag=att_param['attdec_mask_diag'], 
                                       att_mask_flag=att_param['attdec_mask_flag'], 
                                       learnable_scale=att_param['learnable_scale_dec'], 
                                       store_att=store_att,  norm=norm, prenorm=prenorm, activation=activation))
        layers.append(nn.LayerNorm(d_model))
        self.PretrainingDecoder = torch.nn.Sequential(*layers)
        
        #============ Classif Head ============#
        self.GAP = nn.AdaptiveAvgPool1d(1)
        if self.instance_norm:
            self.LinearHead = nn.Linear(d_model*2, nb_class, bias=True)
        else:
            self.LinearHead = nn.Linear(d_model, nb_class, bias=True)
        self.Dropout_Head = (nn.Dropout(dp_rate))

        self.embed_stats = nn.Linear(2, d_model)

        #============ Initializing weights ============#        
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
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
            
    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def subseq_random_masking(self, x):
        N, L, D = x.shape # batch, length, dim
        
        mask = np.ones(L, dtype=bool)
        p_m = 1 / self.mask_mean_length
        p_u = p_m * self.mask_ratio / (1 - self.mask_ratio) 
        p = [p_m, p_u]
        
        state = int(np.random.rand() > self.mask_ratio)
        for i in range(L):
            mask[i] = state
            if np.random.rand() < p[state]:
                state = 1 - state

        mask = torch.Tensor(mask).int().to(x.device)
        
        ids_keep    = torch.nonzero(mask, as_tuple=True)[0]
        ids_removed = torch.nonzero(~mask.bool(), as_tuple=True)[0]
        ids_shuffle = torch.cat((ids_keep, ids_removed))
        ids_restore = torch.argsort(ids_shuffle)
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(0).unsqueeze(-1).repeat(N, 1, D))
        
        return x_masked, ((~mask.bool()).int()).unsqueeze(0).repeat(N, 1), ids_restore.unsqueeze(0).repeat(N, 1)


    def forward_encoder(self, x) -> torch.Tensor:
        # Add Positional Encoding (if any)
        if self.PosEncoding_Encoder is not None:
            x = x + self.PosEncoding_Encoder(x)
            
        # Masking
        if self.mode=="pretraining" or self.mode=="output_mask":
            if self.masking_type=='subsequences':
                x, mask, ids_restore = self.subseq_random_masking(x)
            else:
                x, mask, ids_restore = self.random_masking(x)
        else:
            mask = None
            ids_restore = None

        # Transformer Block
        x = self.EncoderBlock(x)
            
        return x, mask, ids_restore
    
    def forward_decoder_pretraining(self, x, ids_restore) -> torch.Tensor:
        # Add Mask Token
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Add Positional Encoding (if any)
        if self.PosEncoding_Decoder is not None:
            x = x + self.PosEncoding_Decoder(x)

        # Transformer Block
        x = self.PretrainingDecoder(x)
            
        return x

    def forward_loss(self, x_input, x, mask):
        """
        x_input: [N, M, L]
        x: [N, c_reconstruct, L]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        mask = mask.int()
        loss = (x_input[:, :x.shape[1], :] - x) ** 2
        loss = loss.permute(0, 2, 1).mean(dim=-1) # mean loss for each time step
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed tokens

        return loss
    
    def forward(self, x) -> torch.Tensor:
        if self.loss_in_model:
            x_input = x.clone()

        if self.instance_norm:
            self.inst_mean = torch.mean(x[:, :1, :], dim=-1, keepdim=True)
            self.inst_std  = torch.sqrt(torch.var(x[:, :1, :], dim=-1, keepdim=True, unbiased=False) + 1e-5)

            # Instance Norm Agg.
            x[:, :1, :] = (x[:, :1, :] - self.inst_mean) / self.inst_std
        
        # Embedding Block
        x = self.EmbedBlock(x)
        
        # Forward Encoder
        x, mask, ids_restore = self.forward_encoder(x)
        
        # Forward Decoder
        if self.mode=="pretraining" or self.mode=="output_mask":
            x = self.forward_decoder_pretraining(x, ids_restore)

            # Inverse Transform
            x = x.permute(0, 2, 1)

            if self.instance_norm:
                x = x * self.inst_std + self.inst_mean

            x = self.PredHead(x)

            if self.training:
                if self.loss_in_model:
                    loss = self.forward_loss(x_input, x, mask)

                    return x, loss
                else:
                    return x
            else:
                return x, mask if self.mode=="output_mask" else x
                    
        else:
            x = self.GAP(x.permute(0, 2, 1)).flatten(start_dim=1) # B L D -> B D (using GAP)

            if self.instance_norm: 
                stats_proj = F.relu(self.embed_stats(torch.cat([self.inst_mean, self.inst_std], dim=1).flatten(start_dim=1))) # B 1 2 -> B D
                x = torch.cat([x, stats_proj], dim=1)
            
            x = self.Dropout_Head(self.LinearHead(x))
                      
            return x