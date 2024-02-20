# Modified based on https://github.com/XiYe20/VPTR/blob/main/model/VPTR_modules.py

import torch.nn as nn
from .ResNetAutoEncoder import ResnetEncoder, ResnetDecoder
from .VidHRFormer import VidHRFormerFAR
from floodsformer.utils.position_encoding import PositionEmbeddding1D, PositionEmbeddding2D
import functools

class VPTREnc(nn.Module):
    def __init__(self, img_channels, feat_dim = 528, n_downsampling = 3, padding_type = 'reflect'):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_downsampling = n_downsampling
        ngf = 64 if n_downsampling < 5 and feat_dim != 256 and feat_dim != 512 else 32

        self.encoder = ResnetEncoder(
            input_nc = img_channels, 
            ngf = ngf, 
            out_dim = feat_dim, 
            n_downsampling = n_downsampling, 
            padding_type = padding_type
        )
        
    def forward(self, x):
        """
        Args:
            x (tensor): input maps to the encoder (N, T, img_channels, H, W).
        Returns:
            x (tensor): output feature maps of the encoder (N, T, embed_dim, encH, encW).
        """
        N, T, _, _, _ = x.shape
        x = self.encoder(x.flatten(0, 1))
        _, C, H, W = x.shape

        return x.reshape(N, T, C, H, W)

class VPTRDec(nn.Module):
    def __init__(self, img_channels, feat_dim = 528, n_downsampling = 3, out_layer = 'Sigmoid', padding_type = 'reflect'):
        super().__init__()
        self.n_downsampling = n_downsampling
        self.feat_dim = feat_dim
        ngf = 64 if n_downsampling < 5 and feat_dim != 256 and feat_dim != 512  else 32

        self.decoder = ResnetDecoder(
            output_nc = img_channels, 
            ngf = ngf, 
            feat_dim = feat_dim, 
            n_downsampling = n_downsampling, 
            out_layer = out_layer, 
            padding_type = padding_type
        )

    def forward(self, x):
        """
        Args:
            x (tensor): input feature maps of the decoder (N, T, embed_dim, encH, encW)
        Returns:
            x (tensor): output maps of the decoder (N, T, img_channels, H, W)
        """
        N, T, _, _, _ = x.shape
        x = self.decoder(x.flatten(0, 1))
        _, C, H, W = x.shape

        return x.reshape(N, T, C, H, W)

class VPTRDisc(nn.Module):
    """
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    Defines a PatchGAN discriminator
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """
        Args: (N*T, C, H, W).
        Returns: (N*T, 1, h, w).
        """
        return self.model(input)

class VPTRFormerFAR(nn.Module):
    def __init__(
            self, 
            num_past_frames, 
            encH=8,
            encW=8, 
            d_model=512, 
            nhead=8, 
            num_encoder_layers=12, 
            dropout=0.1, 
            window_size=4, 
            Spatial_FFN_hidden_ratio=4,
            rpe=True
        ):
        """
        Fully autoregressive (FAR) VPTR module.
        Args:
            num_past_frames (int): in FS implementation corresponds to the number of input frames (I)
            encH (int): height of compressed maps.
            encW (int): width of compressed maps.
            d_model (int): latent feature dimensionality.
            nhead (int): transformer multi-head attention.
            num_encoder_layers (int): number of Transformer layers.
            dropout (float): dropout rate.
            window_size (int): spatial window size.
            Spatial_FFN_hidden_ratio (int): ratio between hidden_features and in_features in MLP.
            rpe (bool): relative position embedding: https://arxiv.org/abs/2212.06026.
        """
        super().__init__()
        self.num_past_frames = num_past_frames   # corresponds to the number of input frames (I)
        self.num_future_frames = 1     # always equal to 1 for the FAR scheme
        self.nhead = nhead
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        self.dropout = dropout
        self.window_size = window_size
        self.Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio

        self.transformer = VidHRFormerFAR(
            (d_model, encH, encW), 
            num_encoder_layers, 
            num_past_frames, 
            self.num_future_frames,
            d_model, 
            nhead, 
            window_size = window_size, 
            dropout = dropout, 
            drop_path = dropout, 
            Spatial_FFN_hidden_ratio = Spatial_FFN_hidden_ratio, 
            dim_feedforward = self.d_model * Spatial_FFN_hidden_ratio,
            rpe=rpe
        )
        
        # Init all the pos_embed
        T = num_past_frames + self.num_future_frames  # equal to I + 1
        pos1d = PositionEmbeddding1D()
        temporal_pos = pos1d(L = T, N = 1, E = self.d_model)[:, 0, :]
        self.register_buffer('temporal_pos', temporal_pos)
        
        pos2d = PositionEmbeddding2D()
        lw_pos = pos2d(N = 1, E = self.d_model, H = window_size, W = window_size)[0, ...].permute(1, 2, 0)
        self.register_buffer('lw_pos', lw_pos)

        self._reset_parameters()
        
    def forward(self, input_feats):
        """
        Args:
            input_feats (tensor): input features maps of the Transformer (N, T, embed_dim, encH, encW).
        Returns:
            (tensor): output features maps of the Transformer (N, T, embed_dim, encH, encW).
        """
        return self.transformer(input_feats, self.lw_pos, self.temporal_pos)
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
