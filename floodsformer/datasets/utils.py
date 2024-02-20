#!/usr/bin/env python3

from floodsformer.utils import logg
from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np
from floodsformer.utils.preprocessing import Preprocessing

logger = logg.get_logger(__name__)

def DataPartitions(
        total_frames,
        root, 
        max_depth,
        zero_depth,
        partial=None, 
        shuffle_folder=False,
        AR_forec=False,
        num_iter_RTforc=1,
    ):
    ''' 
    Creates the DataLoader partitions for the VPTR train/test.
    File format: 
        xxxx.DEP: water depth maps.
        xxxx.INH: initial water depth map.

    Args:
        total_frames (int):  total number of frames in the sequence.
        root (string): path to the folder with the outputs of the Parflood model.
        max_depth (int): expected maximum water depth for all the simulations
                         (used to normalize the input data).
        zero_depth (float): Threshold (m) to consider a cell dry. During the map loading
                            cells with a value lower than this threshold are set to 0.
        partial (float): percentage of portion of dataset (to load partial, lighter chunks).
        shuffle_folder (bool): shuffle the folders of Parflood's output.
        AR_forec (bool): True to create the test dataset used for the autoregressive procedure.
        num_iter_RTforc (int): Number of iterations for the real-time forecasting.
    Returns:
        dataset_partitions[i]:
            - [0]: list where each element rappresent a string. e.g. "2-12" where 2 is the
                   index of the output folder and 12 is the index of the first map of the sequence.
            - [1]: list where each element rappresent an output folder:
                   - [0]: name of the output folder -> e.g. "output_AAAA_MM_GG_hh_mm".
                   - [1]: list with the name of the DEP and INH files in the folder.
        preprocessing: initialized 'Preprocessing' function.
    '''
    list_sequences = []
    folder_data = []
    preprocessing = None

    outputs = os.listdir(root)
    outputs = [x for x in sorted(outputs) if x.startswith("output_") and os.path.isdir(root + x)]

    if shuffle_folder: # shuffle all the output folders
        np.random.shuffle(outputs)

    if partial is not None:
        outputs = outputs[:int(len(outputs) * partial)]

    assert (len(outputs) > 0), "No output folder found!"

    for n, output in enumerate(outputs):
        dep_filenames = [x for x in os.listdir(os.path.join(root, output)) if x.endswith(".DEP")]  # frames for the specific output folder
        dep_filenames.sort(reverse=False)
        dep_filenames = [x for x in os.listdir(os.path.join(root, output)) if x.endswith(".INH")] + dep_filenames  # add the INH map

        if len(dep_filenames) < total_frames: # Folder discarded because the number of maps contained is lower than the total number of frames in the sequence
            logger.warning("Folder '{}' discarded! Number of maps in the folder lower than the total number of frames in the sequence.".format(output))
            folder_data.append((output, []))
        else:
            folder_data.append((output, dep_filenames))

            if AR_forec:
                size = min(num_iter_RTforc, len(dep_filenames) - total_frames + 1)
            else:
                size = len(dep_filenames) - total_frames + 1  # number of sequences for the specific output folder

            for i in range(size):
                list_sequences.append("{}-{}".format(n, i))  # [index of the output folder] - [index of the first map of the sequence]

            # Initialize Preprocessing function.
            if preprocessing is None:
                preprocessing = Preprocessing(
                    filename=os.path.join(root, output, dep_filenames[1]), 
                    max_depth=max_depth,
                    zero_depth=zero_depth,
                )

    dataset_partitions = [list_sequences, folder_data]
    return dataset_partitions, preprocessing

def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to `True` to have the data reshuffled at every epoch.
        cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset, shuffle=shuffle) if cfg.NUM_GPUS > 1 else None

    return sampler

def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None
