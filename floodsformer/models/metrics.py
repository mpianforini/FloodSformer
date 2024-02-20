
""" Functions for computing metrics. """

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torchmetrics import Recall, Precision

def temporal_weight_func(T):
    t = torch.linspace(0, T-1, T)
    beta = np.log(T)/(T-1)
    w = torch.exp(beta * t)

    return w

# ----------------------------------------------------------
# Metrics for train
# ----------------------------------------------------------

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class L1Loss(nn.Module):
    def __init__(self, temporal_weight = None, norm_dim = None):
        """
        Args:
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.temporal_weight = temporal_weight
        self.norm_dim = norm_dim
    
    def __call__(self, gt, pred):
        """
        pred (tensor): predicted maps (B, T, C, H, W)
        gt: target maps (B, T, C, H, W)
        """
        if self.norm_dim is not None:
            gt = F.normalize(gt, p = 2, dim = self.norm_dim)
            pred = F.normalize(pred, p = 2, dim = self.norm_dim)

        se = torch.abs(pred - gt)
        if self.temporal_weight is not None:
            w = self.temporal_weight.to(se.device)
            if len(se.shape) == 5:
                se = se * w[None, :, None, None, None]
            elif len(se.shape) == 6:
                se = se * w[None, :, None, None, None, None] #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
        mse = se.mean()
        return mse

class MSELoss(nn.Module):
    def __init__(self, temporal_weight = None, norm_dim = None):
        """
        Args:
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.temporal_weight = temporal_weight
        self.norm_dim = norm_dim
    
    def __call__(self, gt, pred):
        """
        pred (tensor): predicted maps (B, T, C, H, W)
        gt: target maps (B, T, C, H, W)
        """
        if self.norm_dim is not None:
            gt = F.normalize(gt, p = 2, dim = self.norm_dim)
            pred = F.normalize(pred, p = 2, dim = self.norm_dim)

        se = torch.square(pred - gt)
        if self.temporal_weight is not None:
            w = self.temporal_weight.to(se.device)
            if len(se.shape) == 5:
                se = se * w[None, :, None, None, None]
            elif len(se.shape) == 6:
                se = se * w[None, :, None, None, None, None] #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
        mse = se.mean()
        return mse

class GDL(nn.Module):
    def __init__(self, alpha = 1, temporal_weight = None):
        """
        Args:
            alpha: hyper parameter of GDL loss, float
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.alpha = alpha
        self.temporal_weight = temporal_weight

    def __call__(self, gt, pred):
        """
        pred (tensor): predicted maps (B, T, C, H, W)
        gt: target maps (B, T, C, H, W)
        """
        B, T, _, _, _ = gt.shape

        gt = gt.flatten(0, -4)
        pred = pred.flatten(0, -4)

        term1 = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])       # gt_i1 - gt_i2
        term2 = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])   # pred_i1 - pred_i2
        term3 = torch.abs(gt[:, :, :, :-1] - gt[:, :, :, 1:])       # gt_j1 - gt_j2
        term4 = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])   # pred_j1 - pred_j2

        if self.alpha != 1:
            gdl1 = torch.pow(torch.abs(term1 - term2), self.alpha)
            gdl2 = torch.pow(torch.abs(term3 - term4), self.alpha)
        else:
            gdl1 = torch.abs(term1 - term2)
            gdl2 = torch.abs(term3 - term4)
        
        if self.temporal_weight is not None:
            assert self.temporal_weight.shape[0] == T, "Mismatch between temporal_weight and predicted sequence length"
            w = self.temporal_weight.to(gdl1.device)
            _, C, H, W = gdl1.shape
            _, C2, H2, W2= gdl2.shape
            gdl1 = gdl1.reshape(B, T, C, H, W)
            gdl2 = gdl2.reshape(B, T, C2, H2, W2)
            gdl1 = gdl1 * w[None, :, None, None, None]
            gdl2 = gdl2 * w[None, :, None, None, None]

        gdl1 = gdl1.mean()
        gdl2 = gdl2.mean()
        gdl_loss = gdl1 + gdl2
        
        return gdl_loss

class BiPatchNCE(nn.Module):
    """
    Bidirectional patchwise contrastive loss
    Implemented Based on https://github.com/alexandonian/contrastive-feature-loss/blob/main/models/networks/nce.py
    """
    def __init__(self, N, T, h, w, temperature = 0.07):
        """
        T: number of frames
        N: batch size
        h: feature height
        w: feature width
        temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        
        #mask meaning; 1 for postive pairs, 0 for negative pairs
        mask = torch.eye(h*w).long()
        mask = mask.unsqueeze(0).repeat(N*T, 1, 1).requires_grad_(False) #(N*T, h*w, h*w)
        self.register_buffer('mask', mask)
        self.temperature = temperature

    def forward(self, gt_f, pred_f):
        """
        gt_f: ground truth feature/images, with shape (N, T, C, h, w)
        pred_f: predicted feature/images, with shape (N, T, C, h, w)
        """
        mask = self.mask

        gt_f = rearrange(gt_f, "N T C h w -> (N T) (h w) C")
        pred_f = rearrange(pred_f, "N T C h w -> (N T) (h w) C")

        #direction 1, decompose the matmul to two steps, Stop gradient for the negative pairs
        score1_diag = torch.matmul(gt_f, pred_f.transpose(1, 2)) * mask
        score1_non_diag = torch.matmul(gt_f, pred_f.detach().transpose(1, 2)) * (1.0 - mask)
        score1 = score1_diag + score1_non_diag #(N*T, h*w, h*w)
        score1 = torch.div(score1, self.temperature)
        
        #direction 2
        score2_diag = torch.matmul(pred_f, gt_f.transpose(1, 2)) * mask
        score2_non_diag = torch.matmul(pred_f, gt_f.detach().transpose(1, 2)) * (1.0 - mask)
        score2 = score2_diag + score2_non_diag
        score2 = torch.div(score2, self.temperature)

        target = (mask == 1).int()
        target = target.to(score1.device)
        target.requires_grad = False
        target = target.flatten(0, 1) #(N*T*h*w, h*w)
        target = torch.argmax(target, dim = 1)

        loss1 = nn.CrossEntropyLoss()(score1.flatten(0, 1), target)
        loss2 = nn.CrossEntropyLoss()(score2.flatten(0, 1), target)
        loss = (loss1 + loss2)*0.5

        return loss

class NoamOpt:
    """
    defatult setup from attention is all you need: 
            factor = 2
            optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    Optim wrapper that implements rate.
    """
    def __init__(self, model_size, factor, train_loader, warmup_epochs, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = len(train_loader)*warmup_epochs
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def reset_step(self, init_epoch, train_loader):
        print("!!!!Learning rate warmup warning: If you are resume training, keep the same Batchsize as before!!!!")
        self._step = len(train_loader) * init_epoch

class SSIM_MSE():
    """Layer to compute the SSIM loss between a pair of images
    From Monodepth2
    https://github.com/nianticlabs/monodepth2/blob/ab2a1bf7d45ae53b1ae5a858f4451199f9c213b3/layers.py#L218
    """
    def __init__(self, reduction=None):
        super().__init__()
        self.mu_x_pool   = torch.nn.AvgPool2d(3, 1)
        self.mu_y_pool   = torch.nn.AvgPool2d(3, 1)
        self.sig_x_pool  = torch.nn.AvgPool2d(3, 1)
        self.sig_y_pool  = torch.nn.AvgPool2d(3, 1)
        self.sig_xy_pool = torch.nn.AvgPool2d(3, 1)

        self.refl = torch.nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        self.loss_fn = torch.nn.MSELoss()

    def __call__(self, predictions, targets):
        ''' 
        Args:
            predictions (tensor): predictions from the current batch. Dimension is
                                  (batches x frames x img_channels x height x width).
            targets (tensor): the corresponding targets of the current batch. Dimension is
                              (batches x frames x img_channels x height x width).
        '''
        loss_mse = self.loss_fn(predictions, targets)

        SSIM = 0.0
        # TODO: think how to remove for loop. The operations are in 2D and do not support 5D tensor, thus for loop on batch.
        for x, y in zip(predictions, targets):
            x = self.refl(x)
            y = self.refl(y)

            mu_x = self.mu_x_pool(x)
            mu_y = self.mu_y_pool(y)

            sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
            sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
            sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

            SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
            SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

            SSIM += torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        
        # TODO: proportion between SSIM and MSE still do be tuned, orginal for depth 0.85 SSIM, 0.15 MSE.
        return (0.15 * SSIM.mean()) + (0.85 * loss_mse)


# ----------------------------------------------------------
# Metrics for test
# ----------------------------------------------------------

class RMSE():
    def __init__(self, threshold=0.05, device='cpu'):
        super().__init__()
        '''
        RMSE metric (root-mean-square error). Used ONLY to check the results (not as a metric for training).
        Compute the rmse error for the entire map and for the wet cells (depth > threshold).
        Args:
            threshold (float): minimum water depth (m) to consider a cell wet.
        '''
        self.loss_all = torch.nn.MSELoss().to(device)
        self.loss_wet = torch.nn.MSELoss(reduction='sum').to(device)
        self.threshold = threshold

    def __call__(self, predictions, targets):
        ''' 
        Args:
            predictions (tensor): predictions from the current batch. Map NOT normalized.
                                  Dimension is (batches x frames x img_channels x height x width).
            targets (tensor): the corresponding targets of the current batch. Map NOT normalized.
                              Dimension is (batches x frames x img_channels x height x width).
        Returns:
            rmse_all (float): rmse metric computed considering the entire map (wet and dry cells).
            rmse_wet (float): rmse metric computed considering only the wet cells in predictions and targets maps.
        '''
        ## rmse error for the entire map.
        rmse_all = torch.sqrt(self.loss_all(predictions, targets))

        ## Consider only cells in predictions and targets maps with water depth > threshold.
        wet_cells = torch.maximum(predictions, targets)
        wet_cells = (wet_cells > self.threshold).type(torch.uint8)
        rmse_wet = torch.sqrt(self.loss_wet(predictions * wet_cells, targets * wet_cells) / wet_cells.sum())

        return rmse_all, rmse_wet

class class_metrics():
    def __init__(self, threshold=0.05, multidim_average = "global", device='cpu'):
        super().__init__()
        '''
        Compute precision, recall and F1 metrics.
        Args:
            threshold (float): minimum water depth (m) to consider a cell wet.
            multidim_average (string): should be 'global' or 'samplewise'. Details can be found in:
                                       https://torchmetrics.readthedocs.io/en/stable/classification/recall.html?highlight=recall
        '''
        self.threshold = threshold
        self.rec_metric = Recall(task = "binary", multidim_average = multidim_average).to(device)     # https://torchmetrics.readthedocs.io/en/stable/classification/recall.html?highlight=recall
        self.prec_metric = Precision(task = "binary", multidim_average = multidim_average).to(device) # https://torchmetrics.readthedocs.io/en/stable/classification/precision.html?highlight=precision

    def __call__(self, predictions, targets):
        ''' 
        Args:
            predictions (tensor): predictions from the current batch. Map NOT normalized.
                                  Dimension is (frames x img_channels x height x width).
            targets (tensor): the corresponding targets of the current batch. Map NOT normalized.
                              Dimension is (frames x img_channels x height x width).
        '''
        # 1: water depth in the cell higher than the threshold.
        # 0: water depth in the cell lower than the threshold.
        predictions = (predictions > self.threshold).type(torch.uint8)
        targets = (targets > self.threshold).type(torch.uint8)

        recall = self.rec_metric(predictions, targets)
        precision = self.prec_metric(predictions, targets)
        f1 = 2.0 * (precision * recall) / (precision + recall)

        return torch.stack((precision, recall, f1))
