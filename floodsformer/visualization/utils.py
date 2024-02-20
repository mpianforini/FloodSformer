
""" Print maps as PNG image. """

import matplotlib.pyplot as plt
import matplotlib as mat
import torch
import os
import numpy as np

import floodsformer.utils.logg as logg

logger = logg.get_logger(__name__)
mat.use("Agg") # prevent graphs from appearing on the screen

class map_to_image():
    def __init__(self, cfg, extensions, color='gist_rainbow'):
        '''
        Args:
            cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
            extensions (list): extensions (coordinates) of the maps (m).
            color (string): color scale used for depth maps.
        Returns:
            cmap_depth (ListedColormap): colormap for depth maps.
            cmap_diff (ListedColormap): colormap for difference maps.  
        '''
        self.n_row = 3    # number of row in the subplot
        self.extensions = extensions
        self.threshold = cfg.DATA.WET_DEPTH   # Minimum water depth (m) to consider a cell wet.
        self.print_val_maps = cfg.TRAIN.PRINT_VAL_MAPS   # If True print as PNG file the validation maps
        self.print_test_maps = cfg.FORECAST.PRINT_TEST_MAPS  # If True print as PNG file the test/real-time forecasting maps

        # Set the colormap for the depth maps
        base = mat.colormaps[color].resampled(256)
        newcolors = base(np.linspace(0, 0.75, 512))
        newcolors[-1, :] = np.array([1, 1, 1, 1])  # [255,255,255,1] -> white
        cmap_depth = mat.colors.ListedColormap(newcolors)
        self.cmap_depth = cmap_depth.reversed()

        # Colorbar of difference maps
        if cfg.DATA.DATASET == "DB_Parma":  # Case study 3: Dam-break of the Parma River flood detention reservoir (Italy).
            plt.rcParams['image.origin']='lower' # Flip maps in vertical direction (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
            self.lambda_max = 0.9
            self.img_width = 3.0
            self.UseExpLabel = True
            self.normal_diff = mat.colors.Normalize(vmin=-2.0, vmax=2.0)

            base = mat.colormaps['binary'].resampled(80)  # steps of 0.05 meters
            newcolors = base(np.linspace(0, 1, 80))
            # Set colors
            newcolors[:10, :] = np.array([144/255, 0, 1, 1])   # purple lower than -1.5
            newcolors[10:20, :] = np.array([1, 87/255, 1, 1])  # pink between -1.5 and -1
            newcolors[20:30, :] = np.array([36/255, 0, 192/255, 1])  # dark blu between -1 and -0.5
            newcolors[30:35, :] = np.array([19/255, 137/255, 1, 1])  # blu between -0.5 and -0.25
            newcolors[35:39, :] = np.array([0, 1, 1, 1])  # cyan between -0.25 and -0.05

            newcolors[39:41, :] = np.array([1, 1, 1, 1])  # white between -0.05 and + 0.05

            newcolors[41:45, :] = np.array([128/255, 1, 87/255, 1])  # green between 0.05 and 0.25
            newcolors[45:50, :] = np.array([51/255, 204/255, 102/255, 1])  # dark green between 0.25 and 0.5
            newcolors[50:60, :] = np.array([1, 1, 0, 1])  # yellow between 0.5 and 1
            newcolors[60:70, :] = np.array([1, 128/255, 0, 1])  # orange between 1 and 1.5
            newcolors[70:, :] = np.array([1, 0, 0, 1])  # red greater than 1.5

            self.ticks_label = [-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5]  # limits in the colorbar of the difference map
        elif cfg.DATA.DATASET == "DB_parabolic":  # Case study 1: Dam-break in a parabolic channel
            plt.rcParams['image.origin']='lower' # Flip maps in vertical direction (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
            self.lambda_max = 0.9
            self.img_width = 2.0
            self.UseExpLabel = False
            self.normal_diff = mat.colors.Normalize(vmin=-2.0, vmax=2.0)

            base = mat.colormaps['binary'].resampled(80)  # steps of 0.05 meters
            newcolors = base(np.linspace(0, 1, 80))
            # Set colors
            newcolors[:10, :] = np.array([144/255, 0, 1, 1])   # purple lower than -1.5
            newcolors[10:20, :] = np.array([1, 87/255, 1, 1])  # pink between -1.5 and -1
            newcolors[20:30, :] = np.array([36/255, 0, 192/255, 1])  # dark blu between -1 and -0.5
            newcolors[30:35, :] = np.array([19/255, 137/255, 1, 1])  # blu between -0.5 and -0.25
            newcolors[35:39, :] = np.array([0, 1, 1, 1])  # cyan between -0.25 and -0.05

            newcolors[39:41, :] = np.array([1, 1, 1, 1])  # white between -0.05 and + 0.05

            newcolors[41:45, :] = np.array([128/255, 1, 87/255, 1])  # green between 0.05 and 0.25
            newcolors[45:50, :] = np.array([51/255, 204/255, 102/255, 1])  # dark green between 0.25 and 0.5
            newcolors[50:60, :] = np.array([1, 1, 0, 1])  # yellow between 0.5 and 1
            newcolors[60:70, :] = np.array([1, 128/255, 0, 1])  # orange between 1 and 1.5
            newcolors[70:, :] = np.array([1, 0, 0, 1])  # red greater than 1.5

            self.ticks_label = [-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5]  # limits in the colorbar of the difference map
        else:  # cfg.DATA.DATASET == "DB_reservoir"   Case study 2: Dam-break in a rectangular tank.
            plt.rcParams['image.origin']='upper' # No flip the maps (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
            self.lambda_max = 1.0
            self.img_width = 2.5
            self.UseExpLabel = False
            self.normal_diff = mat.colors.Normalize(vmin=-0.03, vmax=0.03)

            base = mat.colormaps['binary'].resampled(120)  # steps of 0.0005 meters
            newcolors = base(np.linspace(0, 1, 120))
            # Set colors
            newcolors[:20, :] = np.array([144/255, 0, 1, 1])   # purple lower than -0.02
            newcolors[20:40, :] = np.array([1, 87/255, 1, 1])  # pink between -0.02 and -0.01
            newcolors[40:50, :] = np.array([36/255, 0, 192/255, 1])  # dark blu between -0.01 and -0.005
            newcolors[50:55, :] = np.array([19/255, 137/255, 1, 1])  # blu between -0.005 and -0.0025
            newcolors[55:59, :] = np.array([0, 1, 1, 1])  # cyan between -0.0025 and -0.0005

            newcolors[59:61, :] = np.array([1, 1, 1, 1])  # white between -0.0005 and +0.0005

            newcolors[61:65, :] = np.array([128/255, 1, 87/255, 1])  # green between 0.0005 and 0.0025
            newcolors[65:70, :] = np.array([51/255, 204/255, 102/255, 1])  # dark green between 0.0025 and 0.005
            newcolors[70:80, :] = np.array([1, 1, 0, 1])  # yellow between 0.005 and 0.01
            newcolors[80:100, :] = np.array([1, 128/255, 0, 1])  # orange between 0.01 and 0.02
            newcolors[100:, :] = np.array([1, 0, 0, 1])  # red greater than 0.02

            self.ticks_label = [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]  # limits in the colorbar of the difference map

        self.cmap_diff = mat.colors.ListedColormap(newcolors)

    def print_maps(self, gt_past, gt_future, pred_past, pred_future, save_dir, desc, iterXbatch=0):
        '''
        Print past and future frames (prediction and target).
        Args:
            gt_past (tensor): ground truth past frames (N, Tp, C, H, W).
            gt_future (tensor): ground truth future frames (N, Tf, C, H, W).
            pred_past (tensor): predicted past frames (N, Tp-1, C, H, W).
            pred_future (tensor): predicted future frames (N, Tf, C, H, W).
            save_dir (string): path to the directory to store the images.
            desc (string): image filename prefix.
            iterXbatch (int): for the final test is the number of the current iteration
                              multiplied by the batch size; otherwise is 0.
        '''
        gt_past[gt_past < self.threshold] = 0.0
        gt_future[gt_future < self.threshold] = 0.0
        pred_past[pred_past < self.threshold] = 0.0
        pred_future[pred_future < self.threshold] = 0.0

        max_val = torch.max(torch.max(gt_future), torch.max(pred_future))
        normal = mat.colors.Normalize(vmin = 0, vmax = max_val * self.lambda_max)  # for the colorbar
        N, Tp, C, H, W = gt_past.size()
        Tf = gt_future.shape[1]
        seq_len = Tp + Tf

        diff_past = (pred_past - gt_past[:,1:]).to('cpu', non_blocking=False)
        diff_future = (pred_future - gt_future).to('cpu', non_blocking=False)
        pred_past = pred_past.to('cpu', non_blocking=False)
        gt_past = gt_past.to('cpu', non_blocking=False)
        gt_future = gt_future.to('cpu', non_blocking=False)
        pred_future = pred_future.to('cpu', non_blocking=False)

        # Add a dummy temporal frame (of zeros) in front of pred_past and diff_past to obtain Tp frames
        pred_past = torch.cat((torch.zeros(N, 1, C, H, W), pred_past.detach()), dim=1)
        diff_past = torch.cat((torch.zeros(N, 1, C, H, W), diff_past.detach()), dim=1)

        for b in range(N):
            # ------------- Sequence of maps
            fig, axs = plt.subplots(self.n_row, seq_len, sharex=True, sharey=True, figsize=(self.img_width * seq_len, 5 * self.n_row))
            images = []

            # Past frames
            for j in range(Tp):
                # Ground truth
                axs[0,j].title.set_text('gt past t={}'.format(j + 1 - gt_past.shape[1]))
                images.append(axs[0,j].imshow(gt_past[b,j,0,:,:], norm=normal, cmap=self.cmap_depth, extent=self.extensions))

                # Predicted
                axs[1,j].title.set_text('Pred past t={}'.format(j + 1 - gt_past.shape[1]))
                images.append(axs[1,j].imshow(pred_past[b,j,0,:,:], norm=normal, cmap=self.cmap_depth, extent=self.extensions))

                # Differences
                axs[2,j].title.set_text('Diff past t={}'.format(j + 1 - gt_past.shape[1]))
                images.append(axs[2,j].imshow(diff_past[b,j,0,:,:], norm=self.normal_diff, cmap=self.cmap_diff, extent=self.extensions))

            # Future frames
            for j in range(Tf):
                # Ground truth
                axs[0,j+Tp].title.set_text('gt fut t={}'.format(j + 1))
                images.append(axs[0,j+Tp].imshow(gt_future[b,j,0,:,:], norm=normal, cmap=self.cmap_depth, extent=self.extensions))

                # Predicted
                axs[1,j+Tp].title.set_text('Pred fut t={}'.format(j + 1))
                images.append(axs[1,j+Tp].imshow(pred_future[b,j,0,:,:], norm=normal, cmap=self.cmap_depth, extent=self.extensions))

                # Differences
                axs[2,j+Tp].title.set_text('Diff fut t={}'.format(j + 1))
                images.append(axs[2,j+Tp].imshow(diff_future[b,j,0,:,:], norm=self.normal_diff, cmap=self.cmap_diff, extent=self.extensions))

            for i in range(self.n_row - 1):
                fig.colorbar(images[i], ax=axs[i,:], orientation='vertical', location='right', shrink=0.8, label="Water depth (m)")
            fig.colorbar(images[i+1], ax=axs[i+1,:], orientation='vertical', location='right', shrink=0.8, ticks=self.ticks_label, label="Diff water depth (m)")

            if self.UseExpLabel:  # use scientific notation
                plt.ticklabel_format(axis='both', style='scientific', scilimits=(0,0))

            plt.savefig(os.path.join(save_dir, "{}_seq_{}.png".format(desc, b + iterXbatch)))

            plt.close()
            plt.cla()
            plt.clf()

    def print_maps_AE(self, targets, preds, save_dir, desc, cur_iter):
        '''
        Print predicted and target maps. Used for the AutoEncoder model.
        Args:
            targets (tensor): target frames from the current batch (N, T=1, C, H, W).
            preds (tensor): predicted frames from the current batch (N, T=1, C, H, W).
            save_dir (string): path to the directory to store the images.
            desc (string): image filename prefix.
            cur_iter (int): current iteration.
        '''
        targets[targets < self.threshold] = 0.0
        preds[preds < self.threshold] = 0.0
        seq_len = targets.shape[0]
        max_val = torch.max(torch.max(targets), torch.max(preds))
        normal = mat.colors.Normalize(vmin = 0, vmax = max_val * self.lambda_max)  # for the colorbar

        diff = (preds-targets).to('cpu', non_blocking=False)  # Map of the differences between predicted and target frames
        targets = targets.to('cpu', non_blocking=False)
        preds = preds.to('cpu', non_blocking=False)

        images = []

        if seq_len == 1:
            fig, axs = plt.subplots(self.n_row, seq_len, sharex=True, sharey=True, figsize=(3, 4 * self.n_row))
            # Target frames
            axs[0].title.set_text('Target')
            images.append(axs[0].imshow(targets[0,0,0,:,:], norm=normal, cmap=self.cmap_depth, extent=self.extensions))

            # Predicted frames
            axs[1].title.set_text('Pred')
            images.append(axs[1].imshow(preds[0,0,0,:,:], norm=normal, cmap=self.cmap_depth, extent=self.extensions))

            # Differences
            axs[2].title.set_text('Diff')
            images.append(axs[2].imshow(diff[0,0,0,:,:], norm=self.normal_diff, cmap=self.cmap_diff, extent=self.extensions))

            for i in range(self.n_row - 1):
                fig.colorbar(images[i], ax=axs[i], orientation='vertical', location='right', shrink=0.8, label="Water depth (m)")
            fig.colorbar(images[i+1], ax=axs[i+1], orientation='vertical', location='right', shrink=0.8, ticks=self.ticks_label, label="Diff water depth (m)")
        else:
            fig, axs = plt.subplots(self.n_row, seq_len, sharex=True, sharey=True, figsize=(self.img_width * seq_len, 5 * self.n_row))
            for b in range(seq_len):
                # Target frames
                axs[0,b].title.set_text('Target')
                images.append(axs[0,b].imshow(targets[b,0,0,:,:], norm=normal, cmap=self.cmap_depth, extent=self.extensions))

                # Predicted frames
                axs[1,b].title.set_text('Pred')
                images.append(axs[1,b].imshow(preds[b,0,0,:,:], norm=normal, cmap=self.cmap_depth, extent=self.extensions))

                # Differences
                axs[2,b].title.set_text('Diff')
                images.append(axs[2,b].imshow(diff[b,0,0,:,:], norm=self.normal_diff, cmap=self.cmap_diff, extent=self.extensions))

            for i in range(self.n_row - 1):
                fig.colorbar(images[i], ax=axs[i,:], orientation='vertical', location='right', shrink=0.8, label="Water depth (m)")
            fig.colorbar(images[i+1], ax=axs[i+1,:], orientation='vertical', location='right', shrink=0.8, ticks=self.ticks_label, label="Diff water depth (m)")

        if self.UseExpLabel:
            plt.ticklabel_format(axis='both', style='scientific', scilimits=(0,0))

        plt.savefig(os.path.join(save_dir, "{}_iter_{}.png".format(desc, cur_iter)))

        plt.close()
        plt.cla()
        plt.clf()
