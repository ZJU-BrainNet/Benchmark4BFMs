import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from argparse import Namespace


class CCN_SE_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 128
        return args

    @staticmethod
    def clsf_loss_func(args):
        ce_weight = [1.0 for _ in range(args.n_class)]
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.Adam([{'params': model.parameters(), 'lr': args.model_lr}],
                                    betas=(0.9, 0.999), eps=1e-08)

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        se = self.gap(x)  # shape: (B, C, 1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.hsigmoid(se)
        return x * se

class CCN_SE_Model(nn.Module):
    def __init__(self, args):
        super(CCN_SE_Model, self).__init__()
        self.ccn1 = ConvBlock(args.cnn_in_channels, 64, kernel_size=7, pool_size=2, dropout=0.3)
        self.ccn2 = ConvBlock(64, 128, kernel_size=5, pool_size=2, dropout=0.3)
        self.ccn3 = ConvBlock(128, 256, kernel_size=3, pool_size=2, dropout=0.3)
        self.se = SEBlock(256)
        self.flatten = nn.Flatten()
        if args.patch_len > 125:
            self.fc = nn.Linear(256 * (args.seq_len*125 // 8), args.n_class)  # adjust this to match the final feature map size
        else:
            self.fc = nn.Linear(256 * (args.seq_len*args.patch_len // 8), args.n_class)  # adjust this to match the final feature map size

    def forward(self, x):
        x = self.ccn1(x)
        x = self.ccn2(x)
        x = self.ccn3(x)
        x = self.se(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet

        def downsample_eeg(eeg_data, original_fs, target_fs=125):
            import torchaudio
            B, C, T = eeg_data.shape
            eeg_data = eeg_data.view(B * C, T)
            x_down = torchaudio.functional.resample(eeg_data, orig_freq=original_fs, new_freq=target_fs)
            x_down = x_down.view(B, C, -1)
            return x_down
        
        if args.patch_len > 125:
            x = downsample_eeg(x, args.patch_len, 125)

        bsz, ch_num, N = x.shape
        logit = model(x)
        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y