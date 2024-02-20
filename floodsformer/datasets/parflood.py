
import torch
import math
import os
from floodsformer.utils import logg
from .build import DATASET_REGISTRY
from .utils import DataPartitions

logger = logg.get_logger(__name__)

@DATASET_REGISTRY.register()
class Parflood(torch.utils.data.Dataset):
    """ Parflood frames loader. """

    def __init__(self, cfg, split, batch_size):
        """
        Construct the Parflood loader.
        Args:
            cfg (CfgNode): configs. Details can be found in floodsformer/config/defaults.py
            split (str): the split of the data loader. Options include `train`, `val`, and `test`.
            batch_size (int): batch size.
        """
        self.train_mode = cfg.TRAIN.MODE

        if split == "test":  # non autoregressive test
            self.root = cfg.DATA.PATH_TEST_DATASET
            self.future_frames = 1
            self.past_frames = cfg.DATA.NUM_INPUT_FRAMES
        elif split == "AR_forecast":  # autoregressive test (real-time forecasting)
            self.root = cfg.DATA.PATH_TEST_DATASET
            self.future_frames = cfg.FORECAST.NUM_FUTURE_FRAMES
            self.past_frames = cfg.FORECAST.NUM_PAST_FRAMES
        else: # train or validation
            self.root = cfg.DATA.PATH_TRAIN_DATASET
            self.future_frames = 1
            self.past_frames = cfg.DATA.NUM_INPUT_FRAMES

        self.batch_size = batch_size
        self.image_height = cfg.DATA.IMAGE_HEIGHT
        self.image_width = cfg.DATA.IMAGE_WIDTH
        self.in_channels = cfg.DATA.INPUT_CHANNEL_NUM
        self.out_channels = cfg.DATA.OUTPUT_CHANNEL_NUM
        self.zero_depth = cfg.DATA_LOADER.ZERO_DEPTH
        self.max_depth = cfg.DATA.MAX_DEPTH  # expected maximum water depth for all the simulations (used to normalize the input data)
        self.partial = None            # Percentage of portion of dataset (to load partial, lighter chunks)
        self.shuffle_folder = False    # Shuffle the folders of Parflood's output
        self.num_iter_RTforc = cfg.FORECAST.NUM_ITER

    def prepare_data(self, AR_forec=False):
        # Create partitions: read all the output-xxxx folders and save the id of the frames
        # Initialize the preprocessing (to read and convert maps)
        self.dataset_partitions, self.preprocessing = DataPartitions(
            total_frames=self.past_frames + self.future_frames,
            root=self.root,
            max_depth=self.max_depth,
            zero_depth=self.zero_depth,
            partial=self.partial,
            shuffle_folder=self.shuffle_folder,
            AR_forec=AR_forec,
            num_iter_RTforc=self.num_iter_RTforc,
        )

    def get_datapoint(self, output_index, sequence_index):  
        '''
        Generates the sequence of past and future frames.
        Args:
            output_index (int): index of the output folder (each folder contains the output files of a PARFLOOD simulation).
            sequence_index (int): index of the sequence.
        Returns:
            x (tensor): past frames for the specific sequence [Tp, C, H, W].
            y (tensor): future frames for the specific sequence [Tf, C, H, W].
        '''
        deps = None

        for k in range(sequence_index, sequence_index + self.past_frames + self.future_frames):
            frame = self.preprocessing.read_map(os.path.join(self.root, self.dataset_partitions[1][output_index][0], self.dataset_partitions[1][output_index][1][k]))
            assert (self.image_height==frame.shape[0] and self.image_width==frame.shape[1]), "Size of DEP file ({}, {}) != input size ({}, {})".format(frame.shape[0], frame.shape[1], self.image_height, self.image_width)
            # concatenate maps for different istants
            if deps is None:
                deps = torch.Tensor(frame).unsqueeze(0)
            else:
                deps = torch.cat((deps, torch.Tensor(frame).unsqueeze(0)), dim=0)

        # --- X -> past frames
        x = deps[:self.past_frames].unsqueeze(1)  # add the dimension of the channel -> (Tp, C, H, W)

        # --- Y -> target frames (future frames)
        y = deps[self.past_frames:].unsqueeze(1)  # add the dimension of the channel -> (Tf, C, H, W)

        return x, y

    def dataset_info(self, mode, size, drop_last):
        '''
        Print some information about the dataset.
        Args:
            mode (string): options includes "train", "val", "test" and "AR_forecast" mode.
            size (int): number of sequence in the specific dataset.
            drop_last (bool): drop the last batch (when the number of sequences is not divisible by the batch_size)
        '''
        seq_matrices = (self.past_frames * self.in_channels + self.future_frames * self.out_channels)
        if size > 0:
            ram_db_size = (size * seq_matrices * (self.image_height*self.image_width) * 4) / (1024**3)
            batch_gbytes_size = (self.batch_size * seq_matrices * (self.image_height*self.image_width) * 4) / (1024**3)
            if drop_last:
                n_batches = size//self.batch_size
            else: 
                n_batches = math.ceil(size/self.batch_size)
            logger.info("{} dataset created:\t{} sequences\t{} batches\t{:.3f} GB RAM dataset size\t{:.3f} GB RAM batch size.".format(mode, size, n_batches, ram_db_size, batch_gbytes_size))
        else:
            logger.info("{} dataset is empty.".format(mode))

    def get_preprocessing(self):
        ''' Returns preprocessing function (initialized). '''
        return self.preprocessing

    def __len__(self):
        ''' Return the length of dataset. '''
        return len(self.dataset_partitions[0])

    def __getitem__(self, idx):
        ''' 
        Compute the idx in the dataset and return the sequence of frames.
        Args:
            idx (int): the sequence index provided by the pytorch sampler.
        Returns:
            X (tensor): the input frames of sampled from the sequence [Tp, C, H, W].
            Y (tensor): the output frames of sampled from the sequence [Tf, C, H, W].
        '''
        i, j = self.dataset_partitions[0][idx].split("-")
        # i represents the index of the output folder; j represents the index of the sequence in the output folder i

        X, Y = self.get_datapoint(output_index=int(i), sequence_index=int(j))

        return X, Y
