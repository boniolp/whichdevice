import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Layers.Transformer import EncoderLayer
from Models.Layers.ConvLayer import DilatedBlock, InceptionModule

# ======================= TransApp with concatenate embedding =======================#
class TransAppV2(nn.Module):
    def __init__(self, 
                 mode="classif",
                 c_in=1, c_embedding=6, c_out=1, instance_norm=True,
                 n_embed_blocks=1, type_embedding_block='dilated', 
                 kernel_size=3, nb_class=2,
                 n_encoder_layers=2, n_decoder_layers=1,
                 d_model=64, dp_rate=0.2, activation='gelu',
                 pffn_ratio=4, n_head=8, prenorm=True, norm="LayerNorm", store_att=False, attn_dp_rate=0.2, 
                 att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False,
                            'attdec_mask_diag': True, 'attdec_mask_flag': False, 'learnable_scale_dec': False},
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
        self.EmbedBlock = torch.nn.Sequential(*layers)

        self.ProjEmbedding = nn.Conv1d(in_channels=c_embedding, out_channels=d_model//2, kernel_size=1)
        self.ProjDecoder   = nn.Linear(d_model + d_model//2, d_model)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        d_model = d_model + d_model//2
        self.embed_stats = nn.Linear(2, d_model)
        
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
        _, _, D = x.shape  # batch, length, dim
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked

    def forward_encoder(self, x, ids_keep=None) -> torch.Tensor:
            
        # Masking
        if self.mode=="pretraining" or self.mode=="output_mask":
            x = self.mask_data(x, ids_keep)

        # Transformer Block
        x = self.EncoderBlock(x)
            
        return x
    
    def forward_decoder_pretraining(self, x, encoding, ids_restore) -> torch.Tensor:
        # Add Mask Token
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x = torch.cat([x, encoding.permute(0, 2, 1)], dim=2)

        # Transformer Block
        x = self.PretrainingDecoder(x)
            
        return x

    def forward_loss(self, x_input, x, mask):
        """
        x_input: [N, 1, L]
        x: [N, 1, L]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        mask = mask.int()
        loss = (x_input - x) ** 2
        loss = loss.permute(0, 2, 1).mean(dim=-1) # mean loss for each time step
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed tokens

        return loss
    
    def forward(self, x) -> torch.Tensor:
        # Separate load curve and embedding
        encoding = x[:, 1:, :]
        x        = x[:, :1, :]

        if self.loss_in_model:
            x_input = x.clone()

        if self.instance_norm:
            self.inst_mean = torch.mean(x, dim=-1, keepdim=True)
            self.inst_std  = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - self.inst_mean) / self.inst_std

        if self.mode=="pretraining" or self.mode=="output_mask":
            # Get mask
            mask, ids_keep, ids_restore = self.get_mask(x.device, x.shape[0], x.shape[-1])
            # Set to 0 masked value to avoid leak of information using convolution
            x = x * (~(mask.bool())).int().unsqueeze(1)
        else:
            ids_keep = None

        # Embedding Block
        x = self.EmbedBlock(x)
        encoding = self.ProjEmbedding(encoding)
        x = torch.cat([x, encoding], dim=1).permute(0, 2, 1)
        
        # Forward Encoder
        x = self.forward_encoder(x, ids_keep)
        
        # Forward Decoder
        if self.mode=="pretraining" or self.mode=="output_mask":
            x = self.ProjDecoder(x)
            x = self.forward_decoder_pretraining(x, encoding, ids_restore)

            x = x.permute(0, 2, 1)
            # Reverse Instance Normalization
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