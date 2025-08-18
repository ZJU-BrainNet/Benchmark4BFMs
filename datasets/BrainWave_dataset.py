import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from fractions import Fraction
from einops import rearrange

from data_preprocess.utils import _cal_spec, _std_spec


spec_window_secs_dict = {
    'MAYO': 0.25,
    'FNUSA': 0.25,
    'CHBMIT': 0.25,
    'Siena': 0.25,
    'SleepEDFx': 0.29,
    'RepOD': 0.25,
    'UCSD_ON': 0.25,
    'UCSD_OFF': 0.25,
    'ISRUC': 0.25,
    'ADFD': 0.25,
    'ADHD_Adult': 0.25,
    'ADHD_Child': 0.25,
    'Depression_122_BDI': 0.25,
    'Depression_122_STAI': 0.25,
    'Schizophrenia_28': 0.25,
    'MPHCE_mdd': 0.25,
    'MPHCE_state': 0.25,
    'SEED_IV': 0.25,
    'SD_71': 0.25,
    'EEGMat': 0.25,
    'DEAP': 0.25,
    'EEGMMIDB_R': 0.25,
    'EEGMMIDB_I': 0.25,
}

class BrainWave_Dataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, N)
        # y: (seq_num, )
        self.seq_num, self.ch_num, N = x.shape  
        
        # The operation process of this part is relatively slow. 
        # If you need to reuse the data, you can preprocess and store it first. 
        args.seq_len = N // args.patch_len
        ch_len = args.seq_len*args.patch_len
        x = x[:, :, : ch_len]
        x = x.reshape(-1, args.patch_len)
        
        spec_window_secs = spec_window_secs_dict[args.dataset]
        frac_spec_secs = Fraction(str(spec_window_secs))
        truncate_pts = args.sfreq % frac_spec_secs.denominator
        if truncate_pts > 0:
            x = x[:, :-truncate_pts]
        f, t, Sxx = _cal_spec(args.sfreq-truncate_pts, x, spec_window_secs)
        assert np.min(Sxx) > 0
        Sxx = 10 * np.log10(Sxx)
        # _vis_data_spec(data[10], args.patch_secs, sfreq-truncate_pts, t, f, Sxx[10])
        
        # Sxx: (seq_num, ch_num, seq_len, f_size, t_size)
        Sxx = Sxx.reshape(self.seq_num, self.ch_num, args.seq_len, len(f), len(t))
        Sxx = _std_spec(Sxx)        # normalization

        self.Sxx = Sxx.astype(np.float32)
        self.y = y

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)

    def __getitem__(self, index):
        return self.Sxx[index, :, :, :, :], \
               self.y  [index,]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)

