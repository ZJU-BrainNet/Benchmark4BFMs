import torch
from torch import nn, optim
from argparse import Namespace

from data_process.data_info import data_info_dict
from model.BrainWave.model.encoder import BrainWaveEncoder, LinearHead
from model.BrainWave.model.config import ModelArgs
from model.model_config import ModelPathArgs


class BrainWave_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.load_pretrained = True

        args.seq_len = 16
        args.d_model = args.final_dim = 768

        args.time_n_layers = 8
        args.channel_n_layers = 2
        args.n_heads = 16

        args.norm_eps = 1e-7
        args.ff_hidden = args.d_model * 3
        args.drop_prob = 0.1
        args.learnable_mask = False
        args.mask_ratio = 0.4

        return args

    @staticmethod
    def clsf_loss_func(args):
        if args.n_class == 2:
            ce_weight = [0.3, 1.0]
        else: 
            ce_weight = [1.0 for _ in range(args.n_class)]
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        return  optim.AdamW(
            [
                {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.model_lr},
                {'params': list(clsf.parameters()), 'lr': args.clsf_lr}
            ],
            betas=(0.9, 0.95), eps=1e-5,
        )


    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)



class BrainWave(nn.Module):
    def __init__(self, args: Namespace,):
        super(BrainWave, self).__init__()
        
        self.encoder = self.load_pretrained_weights(args=args,
                            state_dict_path=ModelPathArgs.BrainWave_path
                            )
        
    
    @staticmethod
    def load_pretrained_weights(args, state_dict_path):
        BrainWave_encoder = BrainWaveEncoder(args=ModelArgs).to(args.gpu_id)
        # BrainWave_head = LinearHead(in_features=args.cnn_in_channels * ModelArgs.d_model,
        #                             class_num=args.n_class,
        #                             hidden=True,
        #                             bias=True,
        #                             dropout=0.2).to(args.gpu_id)
        
        if args.load_pretrained:
            BrainWave_state_dict = torch.load(state_dict_path, map_location=f'cuda:{args.gpu_id}')
            encoder_state_dict = {}
            for key, value in BrainWave_state_dict.items():
                if 'prediction_head' in key:
                    continue
                new_key = key.replace('backbone.', '')
                encoder_state_dict[new_key] = value
            BrainWave_encoder.load_state_dict(encoder_state_dict)
            
            return BrainWave_encoder

    
    def forward(self, x):
        t, z, _, _ = self.encoder(x, mask=False)
        z = z[:, :, 0]
        # logit = self.head(z)
        
        return z
    

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        Sxx, y = data_packet
        bsz, ch_num, seq_len, f_size, t_size = Sxx.shape
        
        emb = model(Sxx)
        logit = clsf(emb)

        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y