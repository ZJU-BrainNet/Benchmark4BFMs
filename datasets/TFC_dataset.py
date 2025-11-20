import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import fft
from model.TFC.augmentations import DataTransform_FD, DataTransform_TD

class TFC_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, config, x, y, target_dataset_size=64, subset=False):
        super(TFC_Dataset, self).__init__()
        self.training_mode = config.run_mode
        X_train = torch.from_numpy(x)
        y_train = torch.from_numpy(y)
        # config.TSlength_aligned = config.patch_len*config.seq_len
        config.TSlength_aligned = 178
        self.nProcessLoader = config.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        # if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size * 10 #30 #7 # 60*1
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]
            print('Using subset for debugging, the datasize is:', y_train.shape[0])

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        self.x_data_f = torch.fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        self.len = X_train.shape[0]

        """Augmentation"""
        if config.run_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len
    
    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          drop_last=False, pin_memory=True,
                          shuffle=shuffle,)