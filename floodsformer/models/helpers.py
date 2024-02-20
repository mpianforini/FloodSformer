
import os
import torch
import matplotlib.pyplot as plt

class Loss_tuple(object):
    def __init__(self):
        self.train = 0
        self.val = 0

class EarlyStopper:
    ''' Early stopping function. Monitor a validation metric and stop the training when no improvement is observed. '''
    def __init__(self, patience=10, min_delta=0.0):
        ''' 
        Args:
            patience (int): number of events to wait if no improvement and then stop the training.
            min_delta (float): a minimum increase of the loss to increase the counter.
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.best_epoch = 0
        self.counter = 0
        self.min_validation_loss = 1e10

    def __call__(self, validation_loss, epoch):
        ''' 
        Args:
            validation_loss (float): validation loss used to track improvements.
            epoch (int): current epoch.
        Returns:
            (bool): if True stop the training.
        '''
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.best_epoch = epoch
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def init_loss_dict(loss_name_list, history_loss_dict = None):
    loss_dict = {}
    for name in loss_name_list:
        loss_dict[name] = Loss_tuple()
    loss_dict['epochs'] = 0

    if history_loss_dict is not None:
        for k, v in history_loss_dict.items():
            loss_dict[k] = v

        for k, v in loss_dict.items():
            if k not in history_loss_dict:
                lt = Loss_tuple()
                lt.train = [0] * history_loss_dict['epochs']
                lt.val = [0] * history_loss_dict['epochs']
                loss_dict[k] = lt

    return loss_dict

def val_show_sample(
        past_frames, 
        future_frames, 
        pred_frames, 
        save_dir, 
        preprocessing_map,
        print_image,
        rmse_metric,
        iterXbatch
    ):
    """
    Compute metrics and print maps of the VPTR validation procedure.
    Args:
        past_frames (tensor): ground-truth past frames.
        future_frames (tensor): ground-truth future frames.
        pred_frames (tensor): predicted frames.
        save_dir (string): path to the output folder.
        preprocessing_map: function used to revert the map normalization
                           and save maps as Surfer file.
        print_image: function to print images of the target and predicted maps.
        rmse_metric: function to compute the RMSE.
        iterXbatch (int): batch_size * (rank + iter * word_size).
    Returns:
        rmse_all (tensor): RMSE computed for all the cells of the maps.
        rmse_wet (tensor): RMSE computed only for the wet cells.    
    """
    num_pred = future_frames.shape[1]
    pred_future_frames = pred_frames[:, -num_pred:, ...]

    if preprocessing_map is not None:
        future_frames = preprocessing_map.revert_map_normalize(future_frames)
        pred_future_frames = preprocessing_map.revert_map_normalize(pred_future_frames)
    
    # Compute the RMSE metric (only for future frames).
    rmse_all, rmse_wet = rmse_metric(pred_future_frames[:, :, 0, :, :], future_frames[:, :, 0, :, :])
    with open(os.path.join(save_dir,'rmse_stats.txt'), 'a') as f:
        f.write('VAL: RMSE all maps: {:.4f} m. RMSE wet cells: {:.4f} m\n'.format(rmse_all, rmse_wet))

    if print_image.print_val_maps:
        pred_past_frames = pred_frames[:, 0:-num_pred, ...]
        if preprocessing_map is not None:
            past_frames = preprocessing_map.revert_map_normalize(past_frames)
            pred_past_frames = preprocessing_map.revert_map_normalize(pred_past_frames)

        # Save maps as images (higher computational time).
        print_image.print_maps(
            gt_past=past_frames,
            gt_future=future_frames,
            pred_past=pred_past_frames,
            pred_future=pred_future_frames,
            save_dir=save_dir,
            desc='VAL',
            iterXbatch=iterXbatch,
        )

    return rmse_all, rmse_wet

def RealTimeForecasting(
        VPTR_Enc, 
        VPTR_Dec, 
        VPTR_Transformer, 
        sample, 
        save_dir, 
        device,
        preprocessing_map,
        print_image,
        rmse_metric,
        metrics_classif,
        iterXbatch,
        mod_fut_fram,
    ):
    """
    Real-time forecasting (autoregressive procedure) of the FS model.
    Only past frames used as input (no information about future frames needed).
    Args:
        VPTR_Enc: CNN encoder.
        VPTR_Dec: CNN decoder.
        VPTR_Transformer: VPTR Transformer.
        sample (list):
                [0]: tensor of past frames [N, Tp, C, H, W].
                [1]: tensor of future frames [N, Tf, C, H, W].
        save_dir (string): path to the output folder.
        device (device): current device.
        preprocessing_map: function used to revert the map normalization
                           and save maps as Surfer file.
        print_image: function to print images of the target and predicted maps.
        rmse_metric: function to compute the RMSE.
        metrics_classif: function to compute the precision, recall and F1 metrics.
        iterXbatch (int): batch_size * (rank + iter * word_size).
        mod_fut_fram (int): I + 1 - P.
    Returns:
        mean_rmse (tensor): average RMSE of each future frames.
        mean_class_metrics (tensor): average precision, recall and F1 errors.   
    """
    with torch.no_grad():
        past_frames, future_frames = sample
        num_pred = future_frames.shape[1]
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)

        input_feats = VPTR_Enc(past_frames)
        pred_feats = VPTR_Transformer(input_feats)

        if num_pred <= mod_fut_fram:
            for i in range(num_pred-1):
                pred_fut = VPTR_Dec(pred_feats[:, -1:, ...])   # predicted future frame
                pred_fut = VPTR_Enc(pred_fut)                  # predicted future feature
                input_feats = torch.cat([input_feats, pred_fut], dim = 1)
                pred_feats = VPTR_Transformer(input_feats)
            pred_frames = VPTR_Dec(pred_feats)
        else:
            pred_frames = VPTR_Dec(pred_feats)
            for i in range(num_pred-1):
                pred_feat_fut = VPTR_Enc(pred_frames[:, -1:, ...])
                if i >= mod_fut_fram - 1:
                    input_feats = torch.cat([input_feats[:, 1:, ...], pred_feat_fut], dim = 1)
                else:
                    input_feats = torch.cat([input_feats, pred_feat_fut], dim = 1)
                pred_feats = VPTR_Transformer(input_feats)
                pred_frames = torch.cat([pred_frames, VPTR_Dec(pred_feats[:, -1:, ...])], dim=1)

    pred_past_frames = pred_frames[:, 0:-num_pred, ...]
    pred_future_frames = pred_frames[:, -num_pred:, ...]

    if preprocessing_map is not None:
        past_frames = preprocessing_map.revert_map_normalize(past_frames)
        future_frames = preprocessing_map.revert_map_normalize(future_frames)
        pred_past_frames = preprocessing_map.revert_map_normalize(pred_past_frames)
        pred_future_frames = preprocessing_map.revert_map_normalize(pred_future_frames)

    # Compute the RMSE, Precision, Recall and F1 metrics (only for future frames).
    with open(os.path.join(save_dir, 'rmse_stats.txt'), 'a') as f:
        rmse_app = torch.zeros((pred_future_frames.shape[0], num_pred))
        for b in range(pred_future_frames.shape[0]):
            f.write('TEST - Iter {}. RMSE wet (m): '.format(iterXbatch + b))
            for i in range(num_pred):
                _, rmse_wet = rmse_metric(pred_future_frames[b, i, ...], future_frames[b, i, ...])
                f.write('t={}: {:.5f}\t'.format(i + 1, rmse_wet))
                rmse_app[b, i] = rmse_wet

            class_metrics = metrics_classif(pred_future_frames[b, ...], future_frames[b, ...])
            f.write('\n\t\tPrecision: {:.3f}; Recall: {:.3f}; F1: {:.3f}\n'.format(class_metrics[0], class_metrics[1], class_metrics[2]))
            f.write('Iter {}. Mean RMSE wet: {:.4f}\n'.format(iterXbatch + b, torch.mean(rmse_app[b,:])))
    mean_rmse = torch.mean(rmse_app)

    # Plot the RMSE
    for b in range(pred_future_frames.shape[0]):
        xpts = range(1, rmse_app.shape[1] + 1)
        plt.plot(xpts, rmse_app[b, :])
        plt.xlabel('Future frame (-)')
        plt.ylabel('RMSE (m)')
        plt.title('Real-time forecasting - RMSE')
        plt.savefig(os.path.join(save_dir, 'plot_rmse_seq_{}.png'.format(iterXbatch + b)))
        plt.close('all')

    # Save maps in Surfer 6 binary grid file format.
    preprocessing_map.write_maps(
        past=past_frames,
        target=future_frames, 
        preds=pred_future_frames,
        save_dir=save_dir,
        desc='RealTimeForc',
        iterXbatch=iterXbatch,
    )

    # Save maps as images (high computational time).
    if print_image.print_test_maps:
        print_image.print_maps(
            gt_past=past_frames,
            gt_future=future_frames,
            pred_past=pred_past_frames,
            pred_future=pred_future_frames,
            save_dir=save_dir,
            desc='RealTimeForc',
            iterXbatch=iterXbatch,
        )

    return mean_rmse

def test_show_sample(
        VPTR_Enc, 
        VPTR_Dec, 
        VPTR_Transformer, 
        sample, 
        save_dir, 
        device,
        preprocessing_map,
        print_image,
        rmse_metric,
        metrics_classif,
        iterXbatch,
    ):
    """ 
    Test the VPTR module of the FS model.
    Input to the model: past + future frames.
    Args:
        VPTR_Enc: CNN encoder.
        VPTR_Dec: CNN decoder.
        VPTR_Transformer: VPTR Transformer.
        sample (list):
                [0]: tensor of past frames [N, Tp, C, H, W].
                [1]: tensor of future frames [N, Tf, C, H, W].
        save_dir (string): path to the output folder.
        device (device): current device.
        preprocessing_map: function used to revert the map normalization
                           and save maps as Surfer file.
        print_image: function to print images of the target and predicted maps.
        rmse_metric: function to compute the RMSE.
        metrics_classif: function to compute the precision, recall and F1 metrics.
        iterXbatch (int): batch_size * (rank + iter * word_size).
    Returns:
        mean_rmse (tensor): average RMSE of each future frames.
        mean_class_metrics (tensor): average precision, recall and F1 errors.
    """
    with torch.no_grad():
        past_frames, future_frames = sample
        num_pred = future_frames.shape[1]
        past_frames = past_frames.to(device)
        future_frames = future_frames.to(device)

        x = torch.cat([past_frames, future_frames[:, :-1, ...]], dim = 1)
        pred_frames = VPTR_Dec(VPTR_Transformer(VPTR_Enc(x)))

    pred_past_frames = pred_frames[:, 0:-num_pred, ...]
    pred_future_frames = pred_frames[:, -num_pred:, ...]

    if preprocessing_map is not None:
        past_frames = preprocessing_map.revert_map_normalize(past_frames)
        future_frames = preprocessing_map.revert_map_normalize(future_frames)
        pred_past_frames = preprocessing_map.revert_map_normalize(pred_past_frames)
        pred_future_frames = preprocessing_map.revert_map_normalize(pred_future_frames)

    # Compute the RMSE, Precision, Recall and F1 metrics (only for future frames).
    with open(os.path.join(save_dir, 'rmse_stats.txt'), 'a') as f:
        rmse_app = torch.zeros((pred_future_frames.shape[0], num_pred))
        class_metrics_app = torch.zeros((pred_future_frames.shape[0], 3))
        for b in range(pred_future_frames.shape[0]):
            f.write('TEST - Iter {}. RMSE wet (m): '.format(iterXbatch + b))
            for i in range(num_pred):
                _, rmse_wet = rmse_metric(pred_future_frames[b, i, ...], future_frames[b, i, ...])
                f.write('t={}: {:.4f}\t'.format(i + 1, rmse_wet))
                rmse_app[b, i] = rmse_wet

            class_metrics_app[b] = metrics_classif(pred_future_frames[b, ...], future_frames[b, ...])

            f.write('\n\t\tPrecision: {:.3f}; Recall: {:.3f}; F1: {:.3f}\n'.format(class_metrics_app[b,0], class_metrics_app[b,1], class_metrics_app[b,2]))
        mean_rmse = torch.mean(rmse_app, dim=0)
        mean_class_metrics = torch.mean(class_metrics_app, dim=0)

    # Save maps in Surfer grid file format.
    preprocessing_map.write_maps(
        past=past_frames,
        target=future_frames, 
        preds=pred_future_frames,
        save_dir=save_dir,
        desc='TEST',
        iterXbatch=iterXbatch,
    )

    # Save maps as images (high computational time).
    if print_image.print_test_maps:
        print_image.print_maps(
            gt_past=past_frames,
            gt_future=future_frames,
            pred_past=pred_past_frames,
            pred_future=pred_future_frames,
            save_dir=save_dir,
            desc='TEST',
            iterXbatch=iterXbatch,
        )

    return mean_rmse, mean_class_metrics

def test_show_samples_AE(
        VPTR_Enc, 
        VPTR_Dec, 
        gt_frames, 
        save_dir, 
        preprocessing_map, 
        print_image,
        device,
        rmse_metric,
        metrics_classif,
        cur_iter,
        batch_size,
    ):
    """ 
    Test the AE model.
    Args:
        VPTR_Enc: CNN encoder.
        VPTR_Dec: CNN decoder.
        gt_frames (tensor): ground-truth maps.
        save_dir (string): path to the output folder.
        preprocessing_map: function used to revert the map normalization
                           and save maps as Surfer file.
        print_image: function to print images of the target and predicted maps.
        device (device): current device.
        rmse_metric: function to compute the RMSE.
        metrics_classif: function to compute the precision, recall and F1 metrics.
        cur_iter (int): current iteration * word_size + rank.
        batch_size (int): batch size.
    Returns:
        rmse_all (tensor): RMSE computed for all the cells of the maps.
        rmse_wet (tensor): RMSE computed only for the wet cells.  
        clas_metrics (tensor): precision, recall and F1 errors.
    """
    with torch.no_grad():
        gt_frames = gt_frames.to(device)
        rec_frames = VPTR_Dec(VPTR_Enc(gt_frames))

        if preprocessing_map is not None:
            gt_frames = preprocessing_map.revert_map_normalize(gt_frames)
            rec_frames = preprocessing_map.revert_map_normalize(rec_frames)

    # Compute the RMSE, Precision, Recall and F1 metrics.
    rmse_all_app = 0.0
    rmse_wet_app = 0.0
    class_metrics_app = torch.zeros(3, device=device)

    with open(os.path.join(save_dir, 'rmse_stats.txt'), 'a') as f:
        for b in range(gt_frames.shape[0]):
            rmse_all, rmse_wet = rmse_metric(rec_frames[b], gt_frames[b])
            clas_metrics = metrics_classif(rec_frames[b], gt_frames[b])

            f.write('Frame: {}. RMSE all maps: {:.4f} m; RMSE wet cells: {:.4f} m; Precision: {:.3f}; Recall: {:.3f}; F1: {:.3f}\n'  
                .format(
                    cur_iter * batch_size + b, 
                    rmse_all, 
                    rmse_wet,
                    clas_metrics[0],
                    clas_metrics[1],
                    clas_metrics[2],
                )
            )
            rmse_all_app += rmse_all
            rmse_wet_app += rmse_wet
            class_metrics_app += clas_metrics

    # Save maps in Surfer grid file format.
    preprocessing_map.write_maps_AE(
        target=gt_frames,
        preds=rec_frames,
        save_dir=save_dir,
        desc='TEST',
        cur_iter=cur_iter,
    )

    # Save maps as images (high computational time).
    if print_image.print_test_maps:
        print_image.print_maps_AE(
            targets=gt_frames,
            preds=rec_frames,
            save_dir=save_dir,
            desc='TEST',
            cur_iter=cur_iter,
        )

    return rmse_all_app/(gt_frames.shape[0]), rmse_wet_app/(gt_frames.shape[0]), class_metrics_app/(gt_frames.shape[0])

def val_show_samples_AE(
        gt_frames, 
        rec_frames, 
        save_dir, 
        preprocessing_map,
        print_image,
        rmse_metric, 
        metrics_classif,
        cur_iter
    ):
    """
    Compute metrics and print maps of the AE validation procedure.
    Args:
        gt_frames (tensor): ground-truth maps.
        rec_frames (tensor): reconstructed maps.
        save_dir (string): path to the output folder.
        preprocessing_map: function used to revert the map normalization
                           and save maps as Surfer file.
        print_image: function to print images of the target and predicted maps.
        rmse_metric: function to compute the RMSE.
        metrics_classif: function to compute the precision, recall and F1 metrics.
        cur_iter (int): current iteration * word_size + rank.
    Returns:
        rmse_all (tensor): RMSE computed for all the cells of the maps.
        rmse_wet (tensor): RMSE computed only for the wet cells. 
        clas_metrics[2]: F1 error.
    """
    if preprocessing_map is not None:
        gt_frames = preprocessing_map.revert_map_normalize(gt_frames)
        rec_frames = preprocessing_map.revert_map_normalize(rec_frames)

    # Compute the RMSE and F1 metrics.
    rmse_all, rmse_wet = rmse_metric(rec_frames, gt_frames)
    clas_metrics = metrics_classif(rec_frames, gt_frames)

    # Save metrics on txt file
    with open(os.path.join(save_dir, 'rmse_stats.txt'), 'a') as f:
        f.write('VAL iter {}. RMSE all maps: {:.4f} m; RMSE wet cells: {:.4f} m; F1: {:.3f}\n'.format(cur_iter, rmse_all, rmse_wet, clas_metrics[2]))

    if print_image.print_val_maps:
        # Save maps as images (high computational time).
        print_image.print_maps_AE(
            targets=gt_frames,
            preds=rec_frames,
            save_dir=save_dir,
            desc='VAL',
            cur_iter=cur_iter,
        )

    return rmse_all, rmse_wet, clas_metrics[2]