
""" Loss functions. """

import torch

def cal_lossD(
        VPTR_Disc, 
        fake_imgs, 
        real_imgs, 
        lam_gan, 
        gan_loss
    ):
    '''
    Compute the loss for the discriminator.
    Args:
        VPTR_Disc: PatchGAN discriminator.
        fake_imgs (tensor): predicted maps (N, T, C, H, W).
        real_imgs (tensor): ground truth maps (N, T, C, H, W).
        lam_gan (float): coefficient that multiplies the GAN loss.
        gan_loss (loss): GAN loss function.
    Returns:
        loss_D (float): combined GAN loss.
        loss_D_fake (float): GAN loss for the predicted maps.
        loss_D_real (float): GAN loss for the ground truth maps.
    '''
    pred_fake = VPTR_Disc(fake_imgs.detach().flatten(0, 1))
    loss_D_fake = gan_loss(pred_fake, False)
    # Real
    pred_real = VPTR_Disc(real_imgs.flatten(0,1))
    loss_D_real = gan_loss(pred_real, True)
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5 * lam_gan

    return loss_D, loss_D_fake, loss_D_real

def cal_lossT(
        preds, 
        targets, 
        mse_loss, 
        gdl_loss, 
        l1_loss,
        cfg,
    ):
    '''
    Compute the loss for the Transformer.
    Args:
        preds (tensor): predicted maps (N, T, C, H, W).
        targets (tensor): ground truth maps (N, T, C, H, W).
        mse_loss (loss): MSE loss function.
        gdl_loss (loss): GDL loss function.
        l1_loss (loss): L1 loss function.
        cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
    Returns:
        loss_T (float): total loss (weighted sum of the single losses).
        T_GDL_loss (float): GDL loss.
        T_MSE_loss (float): MSE loss.
        loss_T_gan (float): GAN loss.
        T_L1_loss (float): L1 loss.
    '''
    if cfg.FLOODSFORMER.LAM_MSE != 0.0:
        T_MSE_loss = mse_loss(preds, targets)
    else:
        T_MSE_loss = torch.zeros(1, device=targets.device)

    if cfg.FLOODSFORMER.LAM_GDL != 0.0:
        T_GDL_loss = gdl_loss(targets, preds)
    else:
        T_GDL_loss = torch.zeros(1, device=targets.device)

    if cfg.FLOODSFORMER.LAM_L1 != 0.0:
        T_L1_loss = l1_loss(targets, preds)
    else:
        T_L1_loss = torch.zeros(1, device=targets.device)

    loss_T = cfg.FLOODSFORMER.LAM_GDL * T_GDL_loss + cfg.FLOODSFORMER.LAM_MSE * T_MSE_loss + cfg.FLOODSFORMER.LAM_L1 * T_L1_loss
    return loss_T, T_GDL_loss, T_MSE_loss, T_L1_loss

def cal_lossG(
        VPTR_Disc, 
        fake_imgs, 
        real_imgs, 
        gan_loss, 
        mse_loss, 
        gdl_loss, 
        l1_loss,
        cfg,
    ):
    '''
    Compute the loss for the generator (autoencoder).
    Args:
        VPTR_Disc: PatchGAN discriminator.
        fake_imgs (tensor): predicted maps (N, T, C, H, W).
        real_imgs (tensor): ground truth maps (N, T, C, H, W).
        gan_loss (loss): GAN loss function.
        mse_loss (loss): MSE loss function.
        gdl_loss (loss): GDL loss function.
        l1_loss (loss): L1 loss function.
        cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
    Returns:
        loss_G (float): total loss (weighted sum of the single losses).
        loss_G_gan (float): GAN loss.
        AE_MSE_loss (float): MSE loss.
        AE_GDL_loss (float): GDL loss.
        AE_L1_loss (float): L1 loss.
    '''
    if cfg.FLOODSFORMER.LAM_GAN != 0.0:
        pred_fake = VPTR_Disc(fake_imgs.flatten(0, 1))
        loss_G_gan = gan_loss(pred_fake, True)
    else:
        loss_G_gan = torch.zeros(1, device=real_imgs.device)

    if cfg.FLOODSFORMER.LAM_MSE != 0.0:
        AE_MSE_loss = mse_loss(fake_imgs, real_imgs)
    else:
        AE_MSE_loss = torch.zeros(1, device=real_imgs.device)

    if cfg.FLOODSFORMER.LAM_GDL != 0.0:
        AE_GDL_loss = gdl_loss(real_imgs, fake_imgs)
    else:
        AE_GDL_loss = torch.zeros(1, device=real_imgs.device)

    if cfg.FLOODSFORMER.LAM_L1 != 0.0:
        AE_L1_loss = l1_loss(real_imgs, fake_imgs)
    else:
        AE_L1_loss = torch.zeros(1, device=real_imgs.device)

    loss_G = cfg.FLOODSFORMER.LAM_GAN * loss_G_gan + cfg.FLOODSFORMER.LAM_MSE * AE_MSE_loss + cfg.FLOODSFORMER.LAM_GDL * AE_GDL_loss + cfg.FLOODSFORMER.LAM_L1 * AE_L1_loss

    return loss_G, loss_G_gan, AE_MSE_loss, AE_GDL_loss, AE_L1_loss
