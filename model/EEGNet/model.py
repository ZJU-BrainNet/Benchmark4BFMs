import torch
import torch.nn as nn
import torch
from torch import nn, optim
from argparse import Namespace

class EEGNet_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 128
        return args

    @staticmethod
    def clsf_loss_func(args):
        if args.weights is None:
            ce_weight = [1.0 for _ in range(args.n_class)]
        else:
            ce_weight = args.weights
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.Adam([{'params': model.parameters(), 'lr': args.model_lr}],
                                    betas=(0.9, 0.999), eps=1e-08)

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNet(nn.Module):
    """
    EEGNet 暂未加入weight正则化 max norm 1 and 0.25
    """

    def __init__(self, args, patch_len=None, out_dim=None):
        super(EEGNet, self).__init__()
        self.F1 = 16
        self.D = 2
        self.kern = 25
        self.p = 0.25
        self.F2 = self.F1*self.D
        if patch_len == None:
            patch_len = args.patch_len * args.seq_len
        if out_dim == None:
            out_dim = args.n_class
        self.fcin = self.F2 * (patch_len // 32)
        self.dropout = 0.25

        # 第一层 时间卷积
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kern), padding='same')
        self.bn1 = nn.BatchNorm2d(self.F1)

        # 第二层 空间卷积
        self.conv2 = nn.Conv2d(self.F1, self.D * self.F1, (args.cnn_in_channels, 1), padding='valid', groups=self.F1)
        self.bn2 = nn.BatchNorm2d(self.F1 * self.D)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # 第三层 可分离卷积
        self.sconv3 = SeparableConv(self.F1 * self.D, self.F2, (1, 16), padding='same')
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

        self.fc = nn.Linear(self.fcin, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (bsz, 1, C, T)
        
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)

        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.elu(x)
        x = self.pool2(x)
        x = torch.nn.functional.dropout(x, self.p)

        # 第三层
        x = self.sconv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.elu(x)
        x = self.pool3(x)
        x = torch.nn.functional.dropout(x, self.p)

        # 全连接层
        out = x.view(-1, self.fcin)
        out = self.fc(out)
        out = torch.nn.functional.softmax(out, dim=1)
        return out

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        logit = model(x)
        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y
