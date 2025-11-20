import os
import torch
from torch import nn, optim
from argparse import Namespace

from model.CRT.model import CRTMain, TFR_Encoder, MLP
from model.model_config import ModelPathArgs

class CRT_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        import math
        
        def find_max_k_p(N):    
            n = N // 4
            max_p = 0
            max_k = 0
            seq_len = 0
            if N % 256 != 0:
                return max_p, seq_len
            for p in range(4, int(math.sqrt(n)/2)+1):
                if n % (p * p) == 0:
                    k = n // (p * p)
                    max_p = p
                    max_k = k
                    break
            if max_k != 0:
                seq_len = max_k*max_p*4
            return max_p, seq_len
        
        times = args.patch_len*args.seq_len
        patch_len, seq_len = find_max_k_p(times)
        if patch_len == 0:
            exponent = int(math.log2(times))
            times = 2 ** exponent
            patch_len, seq_len = find_max_k_p(times)
        args.patch_len = patch_len
        args.seq_len = seq_len
        args.final_dim = 128
        args.dim = 128
        args.in_dim = 9     # args.cnn_in_channels
        return args


    @staticmethod
    def clsf_loss_func(args):
        if args.n_class != 2:
            return nn.BCEWithLogitsLoss()
        else:
            if args.weights is None:
                ce_weight = [1.0 for _ in range(args.n_class)]
            else:
                ce_weight = args.weights
            return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.Adam([{'params': model.parameters(), 'lr': args.model_lr}, 
                                 {'params': clsf.parameters(), 'lr': args.clsf_lr}],
                                    betas=(0.9, 0.999), eps=1e-08)
        

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode = 'max', factor=0.8, min_lr=1e-8)
         
         
class Model(nn.Module):
    def __init__(self, args: Namespace):
        super(Model, self).__init__()
        self.encoder = TFR_Encoder(seq_len=args.seq_len,
                            patch_len=args.patch_len,
                            dim=args.dim,
                            num_class=args.n_class,
                            in_dim=args.in_dim)
        self.crt = CRTMain(encoder=self.encoder,
                       decoder_dim=args.dim,
                       in_dim=args.in_dim,
                       patch_len=args.patch_len)
        

    def forward(self, x, ssl = False, ratio = 0.5):
        if ssl == False:
            return self.encoder(x)
        return self.crt(x, mask_ratio=ratio)
    

class CRT(nn.Module):
    def __init__(self, args: Namespace):
        super(CRT, self).__init__()
        self.crt = self.load_pretrained_weights(args)
        # self.classifier = MLP(args.dim, args.dim//2, args.n_class)
        

    def forward(self, x):
        return self.crt(x)
        
        
    @staticmethod
    def load_pretrained_weights(args: Namespace):
        pretrained_model_path = os.path.join(ModelPathArgs.CRT_root_path, f'model_{args.seq_len}_{args.patch_len}_{args.in_dim}.pt')
        crt_model = Model(args)
        
        state_dict = torch.load(pretrained_model_path, map_location=f'cuda:{args.gpu_id}')
        new_state_dict = {}
        for name in state_dict.keys():
            if 'classifier.' in name:
                continue
            new_state_dict[name] = state_dict[name]

        crt_model.load_state_dict(new_state_dict, strict=True)
        for param in crt_model.parameters():
            param.requires_grad = False
        return crt_model
    
    
    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        x = x[:, :, : args.patch_len*args.seq_len]

        emb = model(x)
        logit = clsf(emb)

        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y