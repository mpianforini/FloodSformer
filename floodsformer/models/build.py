
""" Model construction functions. """

import torch
from floodsformer.utils.checkpoint import resume_training_AE, resume_train_transformer
from . import VPTREnc, VPTRDec, VPTRDisc, VPTRFormerFAR, init_weights

def build_AE_model(cfg, device, pretrain=True, build_disc=True):
    """
    Build the AutoEncoder model.
    Args:
        cfg (configs): configs. Details can be seen in floodsformer/config/defaults.py
        device (torch.device): the current device.
        pretrain (bool): load pretrained model.
        build_disc (bool): build the VPTR_Disc.
    """

    VPTR_Enc = VPTREnc(
        cfg.DATA.INPUT_CHANNEL_NUM, 
        feat_dim = cfg.FLOODSFORMER.EMBED_DIM, 
        n_downsampling = cfg.FLOODSFORMER.AE_DOWNSAMPLING
    )
    VPTR_Enc = VPTR_Enc.to(device)
    init_weights(VPTR_Enc, 'VPTR_Enc') # Initialize the model.

    VPTR_Dec = VPTRDec(
        cfg.DATA.OUTPUT_CHANNEL_NUM, 
        feat_dim = cfg.FLOODSFORMER.EMBED_DIM, 
        n_downsampling = cfg.FLOODSFORMER.AE_DOWNSAMPLING, 
        out_layer = cfg.FLOODSFORMER.DEC_OUT_LAYER
    )
    VPTR_Dec = VPTR_Dec.to(device)
    init_weights(VPTR_Dec, 'VPTR_Dec') # Initialize the model.

    if build_disc: # build PatchGAN discriminator
        VPTR_Disc = VPTRDisc(
            cfg.DATA.OUTPUT_CHANNEL_NUM, 
            ndf=64, 
            n_layers=3,
            norm_layer=torch.nn.BatchNorm2d
        )
        VPTR_Disc = VPTR_Disc.to(device)
        init_weights(VPTR_Disc, 'VPTR_Disc') # Initialize the model.

        optimizer_G = torch.optim.Adam(params = list(VPTR_Enc.parameters()) + list(VPTR_Dec.parameters()), lr=cfg.SOLVER.BASE_LR, betas = (0.5, 0.999))
        optimizer_D = torch.optim.Adam(params = VPTR_Disc.parameters(), lr=cfg.SOLVER.BASE_LR, betas = (0.5, 0.999))

        # Load pretrained weights.
        if pretrain:
            start_epoch = resume_training_AE(
                module_dict={'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec, 'VPTR_Disc': VPTR_Disc}, 
                optimizer_dict={'optimizer_G': optimizer_G, 'optimizer_D': optimizer_D},
                resume_ckpt=cfg.MODEL.PRETRAINED_AE,
                map_location='cpu' if cfg.NUM_GPUS == 0 else None,
            )
            start_epoch += 1
        else:
            start_epoch = 0
    else:
        VPTR_Disc = None
        optimizer_D = None
        optimizer_G = None
        # Load pretrained weights.
        if pretrain:
            start_epoch = resume_training_AE(
                module_dict={'VPTR_Enc': VPTR_Enc, 'VPTR_Dec': VPTR_Dec}, 
                optimizer_dict={},
                resume_ckpt=cfg.MODEL.PRETRAINED_AE,
                map_location='cpu' if cfg.NUM_GPUS == 0 else None,
            )
            start_epoch += 1
        else:
            start_epoch = 0

    # Use multi-process data parallel model in the multi-gpu setting.
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        VPTR_Enc = torch.nn.parallel.DistributedDataParallel(module=VPTR_Enc, device_ids=[device], output_device=device)
        VPTR_Dec = torch.nn.parallel.DistributedDataParallel(module=VPTR_Dec, device_ids=[device], output_device=device)
        if build_disc:
            VPTR_Disc = torch.nn.parallel.DistributedDataParallel(module=VPTR_Disc, device_ids=[device], output_device=device)

    return VPTR_Enc, VPTR_Dec, VPTR_Disc, optimizer_D, optimizer_G, start_epoch

def build_TS_model(cfg, device, pretrain=True):
    """
    Build the Transformer model.
    Args:
        cfg (configs): configs. Details can be seen in floodsformer/config/defaults.py
        device (torch.device): the current device.
        pretrain (bool): load pretrained model.
    """

    VPTR_Transformer = VPTRFormerFAR(
        num_past_frames=cfg.DATA.NUM_INPUT_FRAMES,
        encH=cfg.DATA.IMAGE_HEIGHT // 2**cfg.FLOODSFORMER.AE_DOWNSAMPLING, 
        encW=cfg.DATA.IMAGE_WIDTH // 2**cfg.FLOODSFORMER.AE_DOWNSAMPLING, 
        d_model=cfg.FLOODSFORMER.EMBED_DIM, 
        nhead=cfg.FLOODSFORMER.NUM_HEADS, 
        num_encoder_layers=cfg.FLOODSFORMER.T_ENC_LAYERS,
        dropout=cfg.FLOODSFORMER.DROPOUT_RATE, 
        window_size=cfg.FLOODSFORMER.WINDOW_SIZE, 
        Spatial_FFN_hidden_ratio=cfg.FLOODSFORMER.MLP_RATIO, 
        rpe=cfg.FLOODSFORMER.RPE # relative position embedding
    )
    VPTR_Transformer = VPTR_Transformer.to(device)

    # Initialize the model.
    init_weights(VPTR_Transformer, 'VPTR_Transformer')

    # Optimizer
    optimizer_T = torch.optim.AdamW(params = VPTR_Transformer.parameters(), lr = cfg.SOLVER.BASE_LR)

    # Load pretrained weights.
    if pretrain:
        start_epoch, num_input_frames = resume_train_transformer(
            module_dict={'VPTR_Transformer': VPTR_Transformer},
            optimizer_dict={'optimizer_T':optimizer_T},
            resume_ckpt=cfg.MODEL.PRETRAINED_VPTR, 
            map_location='cpu' if cfg.NUM_GPUS == 0 else None,
        )
        start_epoch += 1
    else:
        start_epoch = 0
        num_input_frames = None

    # Use multi-process data parallel model in the multi-gpu setting.
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        VPTR_Transformer = torch.nn.parallel.DistributedDataParallel(
            module=VPTR_Transformer, 
            device_ids=[device], 
            output_device=device
        )

    return VPTR_Transformer, optimizer_T, start_epoch, num_input_frames
