
""" Configs. """

# References:
# Code modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/config/defaults.py
# fvcore: https://github.com/facebookresearch/fvcore/tree/main/fvcore

from fvcore.common.config import CfgNode
import os
from datetime import datetime
import torch

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "runs"
if not os.path.isdir(_C.OUTPUT_DIR): os.mkdir(_C.OUTPUT_DIR)

# Result folder.
folders = [x for x in os.listdir(_C.OUTPUT_DIR) if x.startswith("Run")]
_C.RESULTS_PATH = "Run{}_{}".format(
    len(folders) + 1,
    datetime.now().strftime("%d%b%y_%H%M%S")
)

# Path for prediction results file.
_C.SAVE_RESULTS_PATH = os.path.join(_C.OUTPUT_DIR, _C.RESULTS_PATH)

# Path to the checkpoint folder.
_C.CHECKPOINT_PATH = os.path.join(_C.SAVE_RESULTS_PATH, "checkpoints")

# Random seed.
_C.RNG_SEED = 10

# Log period in iters.
_C.LOG_PERIOD = 10

# Distributed backend.
_C.DIST_BACKEND = "gloo"


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# Name of the file used to construct the dataset.
_C.DATA.DATASET_FILE = "parflood"

# Case studies:
#  1) "DB_parabolic": Dam-break in a parabolic channel.
#  2) "DB_reservoir": Dam-break in a rectangular tank.
#  3) "DB_Parma": Dam-break of the Parma River flood detention reservoir (Italy).
_C.DATA.DATASET = ""

# The path to the maps for the train and validation dataset.
_C.DATA.PATH_TRAIN_DATASET = ""

# The path to the maps for the test dataset.
_C.DATA.PATH_TEST_DATASET = ""

# Maximum water depth of the dataset (used to normalize) [float].
_C.DATA.MAX_DEPTH = 16.0

# Total mini-batch size
_C.DATA.BATCH_SIZE = 12

# Number of input (I) frames (for training).
# For AE training automatically set equal to 0.
_C.DATA.NUM_INPUT_FRAMES = 8

# Input frame channel dimension.
_C.DATA.INPUT_CHANNEL_NUM = 1

# Output frame channel dimension.
_C.DATA.OUTPUT_CHANNEL_NUM = 1

# Image height.
_C.DATA.IMAGE_HEIGHT = 448

# Image width.
_C.DATA.IMAGE_WIDTH = 256

# Size (%) of the validation dataset [float].
_C.DATA.VAL_SIZE = 0.05

# Minimum water depth (m) to consider a cell wet. Used to compute metrics and print maps.
_C.DATA.WET_DEPTH = 0.05


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Model to train: `train_AE`, `train_VPTR`.
_C.TRAIN.MODE = ""

# Evaluate model on validation dataset every eval_period epochs.
_C.TRAIN.EVAL_PERIOD = 5

# If True print as PNG file and save as Surfer grid the VALIDATION maps. Else save
# the maps only as Surfer grid. (If True the computational time of the training increases).
_C.TRAIN.PRINT_VAL_MAPS = False

# Save model checkpoint every checkpoint period epochs. Only for AE model.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# 'patience' of the early stopping function. Only for the VPTR training.
# Number of events to wait if no improvement and then stop the training.
# The number of epochs the function waits are 'TRAIN.EVAL_PERIOD' * 'TRAIN.PATIENCE'.
_C.TRAIN.PATIENCE = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True


# ---------------------------------------------------------------------------- #
# Real-time forecasting options
# ---------------------------------------------------------------------------- #
_C.FORECAST = CfgNode()

# If True do real-time forecasting test, else skip.
_C.FORECAST.ENABLE = True

# Number of past (P) frames for the autoregressive procedure.
# It must be lower than _C.DATA.NUM_INPUT_FRAMES
_C.FORECAST.NUM_PAST_FRAMES = 8

# Number of future (F) frames for the autoregressive procedure.
_C.FORECAST.NUM_FUTURE_FRAMES = 60

# Number of iterations for the real-time forecasting.
# At each iteration the first map of the sequence shifts one position (e.g., for
# iterazion 2 the recursive prediction starts from the frame subsequent at the 
# dam-break). Recommended value: 1
_C.FORECAST.NUM_ITER = 1

# If True print as PNG file and save as Surfer grid the TEST/REAL-TIME FORECASTING maps.
# Else save the maps only as Surfer grid. (If True the computational time of 
# the test/real-time forecasting increases).
_C.FORECAST.PRINT_TEST_MAPS = False


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Load pretrained weights.
_C.MODEL.PRETRAIN = True

# Path to the pretrained AutoEncoder model.
_C.MODEL.PRETRAINED_AE = ''

# Path to the pretrained VPTR model.
_C.MODEL.PRETRAINED_VPTR = ''


# -----------------------------------------------------------------------------
# FloodSformer Options
# -----------------------------------------------------------------------------
_C.FLOODSFORMER = CfgNode()

# Spatial window size.
_C.FLOODSFORMER.WINDOW_SIZE = 4

# Downsampling of the AutoEncoder
_C.FLOODSFORMER.AE_DOWNSAMPLING = 4

# Latent feature dimensionality (number of channels of the latent features).
_C.FLOODSFORMER.EMBED_DIM = 512

# Number of Transformer layers.
_C.FLOODSFORMER.T_ENC_LAYERS = 12

# Number of heads of Transformer multi-head attention.
_C.FLOODSFORMER.NUM_HEADS = 8

# Ratio between hidden_features and in_features in MLP.
_C.FLOODSFORMER.MLP_RATIO = 4

# Relative position embedding: https://arxiv.org/abs/2212.06026.
_C.FLOODSFORMER.RPE = True

# Dropout rate.
_C.FLOODSFORMER.DROPOUT_RATE = 0.1

# Coefficient that multiplies the GDL loss.
_C.FLOODSFORMER.LAM_GDL = 0.01

# Coefficient that multiplies the GAN loss (only for autoencoder training).
_C.FLOODSFORMER.LAM_GAN = 0.01

# Coefficient that multiplies the L1 loss.
_C.FLOODSFORMER.LAM_L1 = 0.0

# Coefficient that multiplies the MSE loss.
_C.FLOODSFORMER.LAM_MSE = 1.0

# Decoder output layer. Activation functions available: Sigmoid, Tanh, Hardtanh, ReLU.
_C.FLOODSFORMER.DEC_OUT_LAYER = 'Sigmoid'


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 500

### Learning rate policy.
# Base learning rate.
_C.SOLVER.BASE_LR = 2e-4

# Learning rate policy (see floodsformer/utils/lr_policy.py for more information).
# Available options:
#   - constant: constant learning rate
#   - cosine: cosine learning rate schedule - (required: 'COSINE_END_LR')
#   - CosAnnealWR: cosine annealing schedule with warm restarts - (required: 'COSINE_END_LR', 'RESTART_EPOCH' and 'RESTART_EPOCH_MULTIP')
#   - steps_with_relative_lrs: steps with relative learning rate schedule - (required: 'STEPS', 'LRS')
_C.SOLVER.LR_POLICY = "constant"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 1e-5

# Steps for 'steps_with_relative_lrs' policies (in epochs).
_C.SOLVER.STEPS = []
# Learning rates for 'steps_with_relative_lrs' policies (list of lr).
# The first value is the start learning rate.
# len(cfg.SOLVER.LRS) == len(cfg.SOLVER.STEPS) + 1
_C.SOLVER.LRS = []

## Cosine annealing schedule with warm restarts options
# Number of iterations for the first restart
_C.SOLVER.RESTART_EPOCH = 50
# A factor multiplies self.restart_epoch at each restart (greater than 0)
_C.SOLVER.RESTART_EPOCH_MULTIP = 1.0

## Warm up options
# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0
# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 1e-6


# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 20

# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 0

# Threshold (m) to consider a cell dry. During the map loading cells with a value 
# lower than this threshold are set to 0.
_C.DATA_LOADER.ZERO_DEPTH = 1e-4

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# Path to save and load dataset.
_C.DATA_LOADER.LOAD_DIR = os.path.join(_C.SAVE_RESULTS_PATH, 'dataset_otf.pt')


# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Enable Tensorboard visualization.
_C.TENSORBOARD.ENABLE = True

# Path to directory for tensorboard logs.
_C.TENSORBOARD.LOG_DIR = os.path.join(_C.SAVE_RESULTS_PATH, "tensorboard")


def _assert_and_infer_cfg(cfg):
    """
    Make some assertions.
    Args:
        cfg (CfgNode): configs.
    """
    # Data assertions.
    if cfg.NUM_GPUS:
        assert cfg.DATA.BATCH_SIZE % cfg.NUM_GPUS == 0

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    assert cfg.DATA.DATASET in ["DB_Parma", "DB_reservoir", "DB_parabolic"], "Dataset type ({}) not supported.".format(cfg.DATA.DATASET)

    if torch.cuda.is_available():
        assert cfg.NUM_GPUS <= torch.cuda.device_count(), "Cannot use more GPU devices than available"
    else:
        assert cfg.NUM_GPUS == 0, "Cuda is not available. Please set `NUM_GPUS: 0` for running on CPUs."

    assert cfg.TRAIN.MODE in ['train_AE', 'train_VPTR'], "Model type ({}) not supported.".format(cfg.TRAIN.MODE)

    # Check input, past and future frames.
    if cfg.TRAIN.MODE == "train_AE":
        if cfg.DATA.NUM_INPUT_FRAMES != 0: cfg.DATA.NUM_INPUT_FRAMES = 0
    else: # "train_VPTR"
        assert cfg.DATA.NUM_INPUT_FRAMES > 0, "Number of input frames ({}) must be greater than 0.".format(cfg.DATA.NUM_INPUT_FRAMES)

        if cfg.FORECAST.ENABLE == True:
            assert cfg.FORECAST.NUM_PAST_FRAMES <= cfg.DATA.NUM_INPUT_FRAMES, "Number of past frames ({}) must be equal or lower than the number of input frames ({}).".format(cfg.FORECAST.NUM_PAST_FRAMES, cfg.DATA.NUM_INPUT_FRAMES)
            assert cfg.FORECAST.NUM_FUTURE_FRAMES > 0, "Number of future frames ({}) must be greater than 0.".format(cfg.FORECAST.NUM_FUTURE_FRAMES)
            assert cfg.FORECAST.NUM_PAST_FRAMES > 0, "Number of past frames ({}) must be greater than 0.".format(cfg.FORECAST.NUM_PAST_FRAMES)

    # Check Autoencoder downsampling.
    assert cfg.DATA.IMAGE_HEIGHT % 2**cfg.FLOODSFORMER.AE_DOWNSAMPLING == 0, "IMAGE_HEIGHT ({}) must be a multiple of 2^AE_DOWNSAMPLING={}.".format(cfg.DATA.IMAGE_HEIGHT, 2**cfg.FLOODSFORMER.AE_DOWNSAMPLING)
    assert cfg.DATA.IMAGE_WIDTH % 2**cfg.FLOODSFORMER.AE_DOWNSAMPLING == 0, "IMAGE_WIDTH ({}) must be a multiple of 2^AE_DOWNSAMPLING={}.".format(cfg.DATA.IMAGE_WIDTH, 2**cfg.FLOODSFORMER.AE_DOWNSAMPLING)

    # Learning rate
    assert cfg.SOLVER.LR_POLICY in ['constant', 'cosine', 'steps_with_relative_lrs', 'CosAnnealWR'], "Learning rate policy ({}) not supported.".format(cfg.SOLVER.LR_POLICY)
    if cfg.SOLVER.LR_POLICY == "cosine": assert cfg.SOLVER.COSINE_END_LR < cfg.SOLVER.BASE_LR
    if cfg.SOLVER.LR_POLICY == "steps_with_relative_lrs": assert len(cfg.SOLVER.LRS) == len(cfg.SOLVER.STEPS) + 1, "The number of lr steps must be equal to the number of lr values + 1"
    if cfg.SOLVER.LR_POLICY == 'CosAnnealWR':
        assert cfg.SOLVER.RESTART_EPOCH != 0, "Set a restart epoch != 0"
        assert cfg.SOLVER.WARMUP_EPOCHS <= cfg.SOLVER.RESTART_EPOCH, "Set a warmup epoch lower than the cosine annealing restart epoch"

    # Check if pretrained model exists.
    if cfg.MODEL.PRETRAIN and cfg.TRAIN.MODE == "train_AE":
        assert os.path.isfile(cfg.MODEL.PRETRAINED_AE), "No pretrained AutoEncoder model found at '{}'.".format(cfg.MODEL.PRETRAINED_AE)

    if cfg.TRAIN.MODE == "train_VPTR":
        assert os.path.isfile(cfg.MODEL.PRETRAINED_AE), "No pretrained AutoEncoder model found at '{}'.".format(cfg.MODEL.PRETRAINED_AE)
        if cfg.MODEL.PRETRAIN:
            assert os.path.isfile(cfg.MODEL.PRETRAINED_VPTR), "No pretrained VPTR model found at '{}'.".format(cfg.MODEL.PRETRAINED_VPTR)

    # Check that the dataset folders exist.
    assert os.path.isdir(cfg.DATA.PATH_TRAIN_DATASET), "Train dataset directory '{}' not found.".format(cfg.DATA.PATH_TRAIN_DATASET)
    assert os.path.isdir(cfg.DATA.PATH_TEST_DATASET), "Test dataset directory '{}' not found.".format(cfg.DATA.PATH_TEST_DATASET)

    # Model assertions.
    if cfg.TRAIN.MODE == "train_VPTR":
        assert cfg.FLOODSFORMER.EMBED_DIM % cfg.FLOODSFORMER.NUM_HEADS == 0, "EMBED_DIM ({}) must be divisible by NUM_HEADS ({}).".format(cfg.FLOODSFORMER.EMBED_DIM, cfg.FLOODSFORMER.NUM_HEADS)
    assert cfg.FLOODSFORMER.DEC_OUT_LAYER in ['Sigmoid', 'Tanh', 'Hardtanh', 'ReLU']

    # Create the output folder.
    if not os.path.isdir(cfg.OUTPUT_DIR): os.mkdir(cfg.OUTPUT_DIR)

    # Try to create the results folder. If it already exists change the name.
    # (This can happen if two simulations start at the same time).
    try:
        os.mkdir(cfg.SAVE_RESULTS_PATH)
    except:
        old_name = cfg.RESULTS_PATH
        folders = [x for x in os.listdir(cfg.OUTPUT_DIR) if x.startswith("Run")]
        # New result folder.
        cfg.RESULTS_PATH = "Run{}_{}".format(len(folders) + 1, datetime.now().strftime("%d%b%y_%H%M%S"))
        # New path to the checkpoint folder.
        if old_name in cfg.CHECKPOINT_PATH:
            cfg.CHECKPOINT_PATH = cfg.CHECKPOINT_PATH.replace(old_name, cfg.RESULTS_PATH)
        # New path for prediction results file.
        if old_name in cfg.SAVE_RESULTS_PATH:
            cfg.SAVE_RESULTS_PATH = cfg.SAVE_RESULTS_PATH.replace(old_name, cfg.RESULTS_PATH)
        # New path to the stored dataset.
        if old_name in cfg.DATA_LOADER.LOAD_DIR:
            cfg.DATA_LOADER.LOAD_DIR = cfg.DATA_LOADER.LOAD_DIR.replace(old_name, cfg.RESULTS_PATH)
        # New path to directory for tensorboard logs.
        if old_name in cfg.TENSORBOARD.LOG_DIR:
            cfg.TENSORBOARD.LOG_DIR = cfg.TENSORBOARD.LOG_DIR.replace(old_name, cfg.RESULTS_PATH)
        # Check that the new output folder does not exist.
        assert not os.path.isdir(cfg.SAVE_RESULTS_PATH), 'Folder "{}" already exists.'.format(cfg.SAVE_RESULTS_PATH)
        os.mkdir(cfg.SAVE_RESULTS_PATH)

    return cfg


def get_cfg():
    """ Get a copy of the default config. """
    return _C.clone()
