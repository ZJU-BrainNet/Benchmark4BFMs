import torch
from omegaconf import OmegaConf
from torch import nn
from argparse import Namespace

from data_process.data_info import data_info_dict
from model.BrainBERT.models.masked_tf_model import MaskedTFModel
from model.pre_cnn import ConvNet
from model.model_config import ModelPathArgs


class BrainBERT_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 768
        return args

    @staticmethod
    def clsf_loss_func(args):
        if args.n_class != 2:
            ce_weight = [1.0 for _ in range(args.n_class)]
        else:
            ce_weight = [0.1, 1]
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, model.cnn.parameters()), 'lr': args.clsf_lr},  # a large lr
            {'params': filter(lambda p: p.requires_grad, model.enc.parameters()), 'lr': args.model_lr},
            {'params': list(clsf.parameters()), 'lr': args.clsf_lr}
        ],
            betas=(0.9, 0.95), eps=1e-5,
        )

    @staticmethod
    def scheduler(optimizer):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)




class BrainBERT(nn.Module):
    def __init__(self, args: Namespace,):
        super(BrainBERT, self).__init__()
        in_patch_len = 40
        self.cnn = ConvNet(num_inputs=1, num_channels=[in_patch_len])

        self.enc = self.load_pretrained_weights(args)

    def forward(self, x):
        bsz, ch_num, seq_len, patch_len = x.shape
        x = x.reshape(bsz*ch_num*seq_len, 1, patch_len)
        emb = self.cnn(x)
        emb = torch.mean(emb, dim=-1).reshape(bsz*ch_num, seq_len, -1)
        emb = self.enc.forward(emb, intermediate_rep=True)
        emb = emb.reshape(bsz, ch_num, seq_len, -1)
        emb = torch.mean(emb, dim=2)
        return emb

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        x = x.reshape(bsz, ch_num, -1, args.patch_len)

        emb = model(x)

        logit = clsf(emb)

        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y

    @staticmethod
    def load_pretrained_weights(args, ):
        def _build_model(cfg, gpu_id):
            ckpt_path = cfg.upstream_ckpt
            init_state = torch.load(ckpt_path, map_location=f'cuda:{gpu_id}')
            upstream_cfg = init_state["model_cfg"]

            # model = models.build_model(upstream_cfg)
            model = MaskedTFModel()
            model.build_model(upstream_cfg, )

            model.load_state_dict(init_state['model'])
            return model

        cfg = OmegaConf.create({"upstream_ckpt": ModelPathArgs.BrainBERT_path})
        model = _build_model(cfg, args.gpu_id).to(args.gpu_id)
        return model

