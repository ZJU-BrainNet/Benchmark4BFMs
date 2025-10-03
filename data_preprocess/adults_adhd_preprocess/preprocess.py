import os
import pickle
import sys
sys.path.append('.../Benchmark4BFMs')
from scipy.io import loadmat
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocess.adults_adhd_preprocess.config import PreprocessArgs
from data_preprocess.adults_adhd_preprocess.utils import _merge_data, _generate_groups, _std_data
from data_preprocess.utils import _segment_data, _std_spec

import random

import torch
from mne.io import read_raw_eeglab
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def generate_subject_data(args):
    for group in ['FADHD', 'FC', 'MADHD', 'MC']:
        mat_data = loadmat(os.path.join(args.data_root, f'{group}.mat'))[group][0]
        group_data = np.swapaxes(np.concatenate(list(mat_data), axis=1), axis1=1, axis2=2)
        sub_num = group_data.shape[0]
        for i in range(sub_num):
            data = group_data[i]
            if group == 'FADHD' and i == 6:
                continue
            # data = _std_data(data)    # std
            data = _segment_data(args, args.sfreq, data)
            np.save(os.path.join(args.data_save_dir, f'{group}_{i}_data.npy'), data)
            print(f'data of subject {group}_{i} saved')


def generate_group_data(args):
    groups = _generate_groups(args)
    for i, g in enumerate(groups):
        data, label = _merge_data(args, g)

        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_data.npy'), data)
        np.save(os.path.join(args.data_save_dir, f'group_data/group_{i}_label.npy'), label)
        print(f'data of group {i} saved')


def merge_group_data(args):
    data, label = [], []
    for g_id in range(args.group_num):
        data.append(np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_data.npy')))
        label.append(np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy')))
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_data.npy'), data)
    np.save(os.path.join(args.data_save_dir, f'group_data/all_label.npy'), label)


def count_group_data(args):
    path = os.path.join(args.data_save_dir, f'subject_groups.pkl')
    groups = pickle.load(open(path, 'rb'))
    category = ['control', 'adhd']
    for g_id in range(args.group_num):
        subject_num = len(groups[g_id])
        print(f'group {g_id}: {subject_num} subjects')
        label = np.load(os.path.join(args.data_save_dir, f'group_data/group_{g_id}_label.npy'))
        for i, c in enumerate(category):
            print(f'class {c}: {np.sum(label==i)} samples')



args = PreprocessArgs()
generate_subject_data(args)
generate_group_data(args)
# merge_group_data(args)
# count_group_data(args)