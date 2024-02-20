# Modified based on: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/datasets/build.py

from fvcore.common.registry import Registry
import os
from floodsformer.utils import logg
import torch

logger = logg.get_logger(__name__)

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.utils.data.dataset.Subset` object.
"""

def build_dataset(dataset_file, cfg, split, n, drop_last, batch_size):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_file (str): the name of the file to construct the dataset.
        cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
        split (str): the split of the data loader. Options include "train", "val", "test" and "AR_forecast".
        n (int): number correlated to the "mode" argument: 0 = train, 1 = val, 2 = test.
        drop_last (bool): drop the last batch (when the number of sequences is not divisible by the batch_size)
        batch_size (int): batch size.
    Returns:
        Dataset: a constructed dataset specified by dataset_file.
    """
    # Capitalize the the first letter of the dataset_file.
    dataset_file = dataset_file.capitalize()
    dataset_SWE = DATASET_REGISTRY.get(dataset_file)(cfg, split, batch_size)

    # Load dataset
    if split == "test":
        logger.info("[~] Creating the test dataset...")
        dataset_SWE.prepare_data()
        assert len(dataset_SWE) > 0, "Empty dataset!!"
        dataset_SWE.dataset_info(split, len(dataset_SWE), drop_last)
        return dataset_SWE
    elif split == "AR_forecast":
        logger.info("[~] Creating the real-time forecast dataset...")
        dataset_SWE.prepare_data(AR_forec=True)
        assert len(dataset_SWE) > 0, "Empty dataset!!"
        logger.info("Number of predicted frames: {}".format(cfg.FORECAST.NUM_FUTURE_FRAMES))
        logger.info("Number of iterations: {}".format(min(cfg.FORECAST.NUM_ITER, len(dataset_SWE))))
        logger.info("Batch size: {}".format(batch_size))
        dataset_SWE.dataset_info(split, len(dataset_SWE), drop_last)
        return dataset_SWE
    elif split == "train":
        # Create the training dataset
        logger.info("[~] Creating the train dataset...")
        dataset_SWE.prepare_data()

        # Split dataset in train and val
        len_dat = len(dataset_SWE)
        assert len_dat > 0, "Empty dataset!!"
        val_len = int(len_dat * cfg.DATA.VAL_SIZE)
        train_len = len_dat - val_len
        datasets = torch.utils.data.random_split(dataset_SWE, [train_len, val_len], generator=torch.Generator())

        # Save the dataset
        torch.save(datasets, cfg.DATA_LOADER.LOAD_DIR)
        logger.info("Saved training dataset: {}".format(cfg.DATA_LOADER.LOAD_DIR))
    else: # split == "val"
        # load dataset from saved file
        if os.path.isfile(cfg.DATA_LOADER.LOAD_DIR) == False:
            raise RuntimeError("Dataset not yet saved. Create the training dataset before the validation dataset.")
        datasets = torch.load(cfg.DATA_LOADER.LOAD_DIR)

    # Print some information about the dataset
    dataset_SWE.dataset_info(split, len(datasets[n]), drop_last)

    return datasets[n]
