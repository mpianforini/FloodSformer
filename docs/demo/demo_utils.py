
""" Functions used in the Colab demo. """

import torch
import os
import matplotlib.pyplot as plt
import matplotlib as mat
import numpy as np
import floodsformer.utils.logg as logg
from datetime import datetime, timedelta

from floodsformer.config.defaults import get_cfg
from floodsformer.utils.distributed import init_distributed_training
from floodsformer.models import build_AE_model, build_TS_model, RMSE, class_metrics
from floodsformer.models.helpers import RealTimeForecasting
from floodsformer.utils.misc import get_device, set_seed, gpu_mem_usage
from floodsformer.datasets.loader import construct_loader

logger = logg.get_logger(__name__)

class map_to_image():
    def __init__(self, cfg):
        self.threshold = cfg.DATA.WET_DEPTH
        self.print_val_maps = False
        self.print_test_maps = False

def set_color(cfg, color='gist_rainbow'):
    # Set the colormap for the depth maps
    base = mat.colormaps[color].resampled(256)
    newcolors = base(np.linspace(0, 0.75, 512))
    newcolors[-1, :] = np.array([1, 1, 1, 1])
    cmap_depth = mat.colors.ListedColormap(newcolors)
    cmap_depth = cmap_depth.reversed()

    # Colorbar of difference maps
    if cfg.DATA.DATASET == "DB_Parma":  # Dam-break of the Parma River flood detention reservoir (Italy).
        plt.rcParams['image.origin']='lower'
        lambda_max = 0.9
        img_width = 3.0
        normal_diff = mat.colors.Normalize(vmin=-2.0, vmax=2.0)

        base = mat.colormaps['binary'].resampled(80)
        newcolors = base(np.linspace(0, 1, 80))
        # Set colors
        newcolors[:10, :] = np.array([144/255, 0, 1, 1])
        newcolors[10:20, :] = np.array([1, 87/255, 1, 1])
        newcolors[20:30, :] = np.array([36/255, 0, 192/255, 1])
        newcolors[30:35, :] = np.array([19/255, 137/255, 1, 1])
        newcolors[35:39, :] = np.array([0, 1, 1, 1])
        newcolors[39:41, :] = np.array([1, 1, 1, 1])
        newcolors[41:45, :] = np.array([128/255, 1, 87/255, 1])
        newcolors[45:50, :] = np.array([51/255, 204/255, 102/255, 1])
        newcolors[50:60, :] = np.array([1, 1, 0, 1])
        newcolors[60:70, :] = np.array([1, 128/255, 0, 1])
        newcolors[70:, :] = np.array([1, 0, 0, 1])

        ticks_label = [-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5]
    elif cfg.DATA.DATASET == "DB_parabolic":  # Dam-break in a parabolic channel
        plt.rcParams['image.origin']='lower'
        lambda_max = 0.9
        img_width = 2.0
        normal_diff = mat.colors.Normalize(vmin=-2.0, vmax=2.0)

        base = mat.colormaps['binary'].resampled(80)
        newcolors = base(np.linspace(0, 1, 80))
        # Set colors
        newcolors[:10, :] = np.array([144/255, 0, 1, 1])
        newcolors[10:20, :] = np.array([1, 87/255, 1, 1])
        newcolors[20:30, :] = np.array([36/255, 0, 192/255, 1])
        newcolors[30:35, :] = np.array([19/255, 137/255, 1, 1])
        newcolors[35:39, :] = np.array([0, 1, 1, 1])
        newcolors[39:41, :] = np.array([1, 1, 1, 1])
        newcolors[41:45, :] = np.array([128/255, 1, 87/255, 1])
        newcolors[45:50, :] = np.array([51/255, 204/255, 102/255, 1])
        newcolors[50:60, :] = np.array([1, 1, 0, 1])
        newcolors[60:70, :] = np.array([1, 128/255, 0, 1])
        newcolors[70:, :] = np.array([1, 0, 0, 1])

        ticks_label = [-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5]
    else:  # Dam-break in a rectangular tank.
        plt.rcParams['image.origin']='upper'
        lambda_max = 1.0
        img_width = 2.5
        normal_diff = mat.colors.Normalize(vmin=-0.03, vmax=0.03)

        base = mat.colormaps['binary'].resampled(120)
        newcolors = base(np.linspace(0, 1, 120))
        # Set colors
        newcolors[:20, :] = np.array([144/255, 0, 1, 1])
        newcolors[20:40, :] = np.array([1, 87/255, 1, 1])
        newcolors[40:50, :] = np.array([36/255, 0, 192/255, 1])
        newcolors[50:55, :] = np.array([19/255, 137/255, 1, 1])
        newcolors[55:59, :] = np.array([0, 1, 1, 1])
        newcolors[59:61, :] = np.array([1, 1, 1, 1])
        newcolors[61:65, :] = np.array([128/255, 1, 87/255, 1])
        newcolors[65:70, :] = np.array([51/255, 204/255, 102/255, 1])
        newcolors[70:80, :] = np.array([1, 1, 0, 1])
        newcolors[80:100, :] = np.array([1, 128/255, 0, 1])
        newcolors[100:, :] = np.array([1, 0, 0, 1])

        ticks_label = [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]

    cmap_diff = mat.colors.ListedColormap(newcolors)

    return [lambda_max, img_width], ticks_label, cmap_diff, cmap_depth, normal_diff

def MapToImage(cfg, save_dir, renorm_transform):
    n_row = 3
    n_past = cfg.FORECAST.NUM_PAST_FRAMES
    n_future = cfg.FORECAST.NUM_FUTURE_FRAMES
    print("Past frames: {}".format(n_past))
    print("Future frames: {}".format(n_future))
    maps_path = os.path.join(save_dir, 'RealTimeForc_iter_0')

    dat, ticks_label, cmap_diff, cmap_depth, normal_diff = set_color(cfg)

    past_frames = None
    gt_fut_frames = None
    pred_fut_frames = None

    if n_past > 1:
        P_frames = [-n_past + 1, 0]
    else:
        P_frames = [0]

    for _, n in enumerate(P_frames):
        map_name = "Past_t" + str(n) + ".grd"
        frame = renorm_transform.read_map(os.path.join(maps_path, map_name))
        if past_frames is None:
            past_frames = torch.Tensor(frame).unsqueeze(0)
        else:
            past_frames = torch.cat((past_frames, torch.Tensor(frame).unsqueeze(0)), dim=0)

    if n_future <= 7:
        F_frames = list(range(1, n_future + 1))
    else:
        F_frames = [1, 2]

        if n_past == 1:
            F_frames.append(4)
            F_frames.append(9)
        elif n_past < 8:
            F_frames.append(10 - n_past)

        for i in range(15 - n_past, n_future + 1, 15):
            F_frames.append(i)

    for _, n in enumerate(F_frames):
        map_name = "True_t" + str(n) + ".grd"
        frame = renorm_transform.read_map(os.path.join(maps_path, map_name))
        if gt_fut_frames is None:
            gt_fut_frames = torch.Tensor(frame).unsqueeze(0)
        else:
            gt_fut_frames = torch.cat((gt_fut_frames, torch.Tensor(frame).unsqueeze(0)), dim=0)

        map_name = "Pred_t" + str(n) + ".grd"
        frame = renorm_transform.read_map(os.path.join(maps_path, map_name))
        if pred_fut_frames is None:
            pred_fut_frames = torch.Tensor(frame).unsqueeze(0)
        else:
            pred_fut_frames = torch.cat((pred_fut_frames, torch.Tensor(frame).unsqueeze(0)), dim=0)


    past_frames = renorm_transform.revert_map_normalize(past_frames)
    gt_fut_frames = renorm_transform.revert_map_normalize(gt_fut_frames)
    pred_fut_frames = renorm_transform.revert_map_normalize(pred_fut_frames)

    diff_fut_frames = pred_fut_frames - gt_fut_frames

    past_frames[past_frames < cfg.DATA.WET_DEPTH] = 0.0
    gt_fut_frames[gt_fut_frames < cfg.DATA.WET_DEPTH] = 0.0
    pred_fut_frames[pred_fut_frames < cfg.DATA.WET_DEPTH] = 0.0

    max_val = torch.max(torch.max(gt_fut_frames), torch.max(pred_fut_frames))
    normal = mat.colors.Normalize(vmin = 0, vmax = max_val * dat[0])
    P_frames_num, H, W = past_frames.size()
    F_frames_num = gt_fut_frames.shape[0]
    seq_len = P_frames_num + F_frames_num

    dummy_past = torch.zeros(H, W)

    # ------------- Sequence of maps
    fig, axs = plt.subplots(n_row, seq_len, sharex=True, sharey=True, figsize=(dat[1] * seq_len, 5 * n_row), dpi=400)
    images = []

    # Past frames
    for j in range(P_frames_num):
        # Ground truth
        axs[0,j].title.set_text('gt past t={}'.format(P_frames[j] + n_past))
        images.append(axs[0,j].imshow(past_frames[j,:,:], norm=normal, cmap=cmap_depth, extent=renorm_transform.extensions))

        # Predicted
        #axs[1,j].title.set_text('Pred past t={}'.format(P_frames[j] + n_past))
        images.append(axs[1,j].imshow(dummy_past[:,:], norm=normal, cmap=cmap_depth, extent=renorm_transform.extensions))

        # Differences
        #axs[2,j].title.set_text('Diff past t={}'.format(P_frames[j] + n_past))
        images.append(axs[2,j].imshow(dummy_past[:,:], norm=normal_diff, cmap=cmap_diff, extent=renorm_transform.extensions))

    # Future frames
    for j in range(F_frames_num):
        # Ground truth
        axs[0,j+P_frames_num].title.set_text('gt fut t={}'.format(F_frames[j] + n_past))
        images.append(axs[0,j+P_frames_num].imshow(gt_fut_frames[j,:,:], norm=normal, cmap=cmap_depth, extent=renorm_transform.extensions))

        # Predicted
        axs[1,j+P_frames_num].title.set_text('Pred fut t={}'.format(F_frames[j] + n_past))
        images.append(axs[1,j+P_frames_num].imshow(pred_fut_frames[j,:,:], norm=normal, cmap=cmap_depth, extent=renorm_transform.extensions))

        # Differences
        axs[2,j+P_frames_num].title.set_text('Diff fut t={}'.format(F_frames[j] + n_past))
        images.append(axs[2,j+P_frames_num].imshow(diff_fut_frames[j,:,:], norm=normal_diff, cmap=cmap_diff, extent=renorm_transform.extensions))

    for i in range(n_row - 1):
        fig.colorbar(images[i], ax=axs[i,:], orientation='vertical', location='right', shrink=0.8, label="Water depth (m)")
    fig.colorbar(images[i+1], ax=axs[i+1,:], orientation='vertical', location='right', shrink=0.8, ticks=ticks_label, label="Diff water depth (m)")

    if cfg.DATA.DATASET == "DB_Parma":  # use scientific notation
        plt.ticklabel_format(axis='both', style='scientific', scilimits=(0,0))

    plt.show()
    plt.close()

def set_config(
        cfg_file, 
        past_frames, 
        future_frames, 
        chkpt_AE_dir, 
        chkpt_VPTR_dir,
        dataset_dir,
    ):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.FORECAST.NUM_FUTURE_FRAMES = future_frames
    cfg.FORECAST.NUM_PAST_FRAMES = past_frames
    cfg.MODEL.PRETRAINED_AE = chkpt_AE_dir
    cfg.MODEL.PRETRAINED_VPTR = chkpt_VPTR_dir
    cfg.DATA.PATH_TEST_DATASET = dataset_dir
    cfg.DATA.PATH_TRAIN_DATASET = dataset_dir # not used (only for compatibility)
    assert cfg.TRAIN.MODE == "train_VPTR", "Set cfg.TRAIN.MODE='train_VPTR' for the real-time forecasting procedure."

    return cfg

def run_RTforecast(cfg):
    # Run the real-time forecasting procedure
    init_distributed_training(cfg)
    device = get_device(cfg)
    # Set random seed from configs.
    set_seed(cfg.RNG_SEED)

    # Setup logging format.
    logg.setup_logging(cfg.SAVE_RESULTS_PATH, cfg.NUM_GPUS * cfg.NUM_SHARDS)
    logger.info("### Real-time forecasting ###")

    # Init test dataset
    test_loader = construct_loader(cfg, "AR_forecast")

    renorm_transform = test_loader.dataset.get_preprocessing()
    print_image = map_to_image(cfg)

    # Init Models and Optimizer
    VPTR_Enc, VPTR_Dec, _, _, _, _ = build_AE_model(cfg, device, pretrain=True, build_disc=False) # load AE pretrained model always

    # For the real-time forecasting application load always the VPTR model.
    assert os.path.isfile(cfg.MODEL.PRETRAINED_VPTR), "No pretrained VPTR model found at '{}'.".format(cfg.MODEL.PRETRAINED_VPTR)
    VPTR_Transformer, _, epoch, num_input_frames = build_TS_model(cfg, device, pretrain=True)

    if num_input_frames is not None:
        # For real-time forecasting the model must be equal to the trained one (no interpolation of parameters is allowed).
        assert num_input_frames == cfg.DATA.NUM_INPUT_FRAMES, "Load a pretrained model with number of input frames = {} but cfg.DATA.NUM_INPUT_FRAMES = {}!".format(num_input_frames, cfg.DATA.NUM_INPUT_FRAMES)

    VPTR_Enc = VPTR_Enc.eval()
    VPTR_Dec = VPTR_Dec.eval()
    VPTR_Transformer = VPTR_Transformer.eval()

    rmse_metric = RMSE(threshold=cfg.DATA.WET_DEPTH, device=device)
    metrics_classif = class_metrics(threshold=cfg.DATA.WET_DEPTH, device=device)

    # Real-time forecasting
    save_dir = os.path.join(cfg.SAVE_RESULTS_PATH, "R-TForc_e{}_I{}_P{}_F{}".format(epoch - 1, cfg.DATA.NUM_INPUT_FRAMES, cfg.FORECAST.NUM_PAST_FRAMES, cfg.FORECAST.NUM_FUTURE_FRAMES))
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    mean_rmse_app = None
    test_start = datetime.now()
    mod_fut_fram = cfg.DATA.NUM_INPUT_FRAMES + 1 - cfg.FORECAST.NUM_PAST_FRAMES

    logger.info("Run real-time forecasting")
    for iter, sample in enumerate(test_loader):
        mean_rmse = RealTimeForecasting(
            VPTR_Enc,
            VPTR_Dec,
            VPTR_Transformer,
            sample,
            save_dir,
            device,
            renorm_transform,
            print_image,
            rmse_metric,
            metrics_classif,
            iterXbatch=iter*test_loader.batch_size,
            mod_fut_fram=mod_fut_fram,
        )
        if mean_rmse_app is None:
            mean_rmse_app = mean_rmse.unsqueeze(0)
        else:
            mean_rmse_app = torch.cat((mean_rmse_app, mean_rmse.unsqueeze(0)), dim=0)

        if (iter + 1) * test_loader.batch_size >= cfg.FORECAST.NUM_ITER:
            break

    mean_rmse_app = torch.mean(mean_rmse_app)
    test_time = datetime.now() - test_start

    logger.info(
        "End of real-time forecasting - Duration: {}. Mean RMSE: {:.4f} m; Max_gpu_mem: {:.2f}/{:.2f}G."
        .format(
            timedelta(seconds=int(test_time.total_seconds())),
            mean_rmse_app,
            gpu_mem_usage()[0],
            gpu_mem_usage()[1]
        )
    )

    return save_dir, renorm_transform