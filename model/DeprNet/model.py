import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from torch import optim


class DeprNet_Trainer:
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
    

class DeprNet(nn.Module):
    def __init__(self, args: Namespace):
        super(DeprNet, self).__init__()
        self.path_len = args.patch_len
        self.seq_len = args.seq_len
        N = args.patch_len*args.seq_len
        dim_num = (((((N-4)//4-4)//4-4)//4-2)//4-1)//4*args.cnn_in_channels*32

        self.cov0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),)
        
        self.cov1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),)
        
        self.cov2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),)
        
        self.cov3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),)

        self.cov4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_num, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, args.n_class),
            nn.Softmax(dim=1))


    def forward(self, x):
        bsz, ch_num, N = x.shape
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # (batch, 3, C, N)
        x = self.cov0(x)
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.cov3(x)
        x = self.cov4(x)
        x = self.cls(x)
        return x

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        logit = model(x)
        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y