# Modified based on https://github.com/XiYe20/VPTR/blob/main/model/VidHRFormer.py

import torch.nn as nn
import torch.nn.functional as F
from .VidHRFormer_modules import VidHRFormerEncoder, VidHRFormerBlockEnc

class VidHRFormerFAR(nn.Module):
    def __init__(
            self, 
            in_feat_shape, 
            num_encoder_layer, 
            num_past_frames, 
            num_future_frames,
            embed_dim, 
            num_heads, 
            window_size = 7, 
            dropout = 0., 
            drop_path = 0., 
            Spatial_FFN_hidden_ratio = 4, 
            dim_feedforward = 512, 
            rpe = True
        ):
        super().__init__()
        self.in_C, self.H, self.W = in_feat_shape
        self.embed_dim = embed_dim
        #self.conv_proj = nn.Identity() if self.in_C == self.embed_dim else self.feat_proj()
        #self.conv_proj_rev = nn.Identity() if self.in_C == self.embed_dim else self.feat_proj_rev()

        self.num_encoder_layer = num_encoder_layer
        self.num_heads = num_heads

        self.encoder = VidHRFormerEncoder(
                VidHRFormerBlockEnc(
                        self.H, 
                        self.W, 
                        embed_dim, 
                        num_heads, 
                        window_size, 
                        dropout, 
                        drop_path, 
                        Spatial_FFN_hidden_ratio, 
                        dim_feedforward, 
                        far = True, 
                        rpe=rpe
                ), 
                num_encoder_layer, 
                nn.LayerNorm(embed_dim)
        )

    def forward(self, input_feat, local_window_pos_embed, temporal_pos_embed):
        """
        Args:
            input_feats (tensor): input features maps of the Transformer (N, T, embed_dim, encH, encW).
            local_window_pos_embed (tensor): spatial positional embedding (window_size, window_size, embed_dim).
            temporal_pos_embed (tensor): temporal positional embedding (Tp+Tf, embed_dim).
        Return:
            out: (N, Tf, H, W, embed_dim), for the next layer query_pos init
            out_proj: (N, Tf, in_C, H, W), final output feature for the decoder
            memory: (N, Tp, H, W, embed_dim)
        """
        _, T, _, _, _ = input_feat.shape
        input_feat = input_feat.permute(0, 1, 3, 4, 2)  # (N,T,embed_dim,encH,encW) -> (N,T,encH,encW,embed_dim)
        pred = self.encoder(input_feat, local_window_pos_embed, temporal_pos_embed[0:T, ...])
        pred = F.relu_(pred.permute(0, 1, 4, 2, 3))

        return pred