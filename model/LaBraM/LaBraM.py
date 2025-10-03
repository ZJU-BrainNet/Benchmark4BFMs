from collections import OrderedDict
import librosa
import torch
from torch import nn, optim
from argparse import Namespace
from einops import rearrange

from data_process.data_info import data_info_dict
from model.pre_cnn import ConvNet
from model.LaBraM import utils
from model.model_config import ModelPathArgs

class LaBraM_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):

        args._model = 'labram_base_patch200_200'
        args.nb_classes = 2
        args.drop = 0.0
        args.drop_path = 0.1
        args.attn_drop_rate = 0.0

        args.use_mean_pooling = True
        args.init_scale = 0.001
        args.rel_pos_bias = True
        args.abs_pos_emb = False
        args.layer_scale_init_value = 0.1
        args.qkv_bias = True

        args.finetune = ModelPathArgs.LaBraM_path

        args.model_key = 'model|module'
        args.model_filter_name = 'gzp'
        args.model_prefix = ''

        args.final_dim = 1024
        args.tune_a_part = True

        return args

    @staticmethod
    def clsf_loss_func(args):
        ce_weight = [1.2 for _ in range(args.n_class - 1)]
        ce_weight.append(1.0)
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, model.model_clsf.blocks.parameters()), 'lr': args.model_lr},
            {'params': filter(lambda p: p.requires_grad, model.model_clsf.head.parameters()),   'lr': args.clsf_lr},
        ],
            betas=(0.9, 0.99), eps=1e-8,
        )

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)







class LaBraM(nn.Module):
    def __init__(self, args: Namespace,):
        super(LaBraM, self).__init__()
        self.model_clsf = self.load_pretrained_weights(args)
        self.model_clsf = self.freeze_part(args, self.model_clsf)

    def forward(self, x):
        logit = self.model_clsf(x)
        return logit

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, N = x.shape
        
        # resampling
        if args.sfreq != 200:
            x = x.reshape(bsz*ch_num, -1)
            x = x.to('cpu').numpy()
            x = librosa.resample(x, orig_sr=args.sfreq, target_sr=200)
            x = torch.from_numpy(x).to(f'cuda:{args.gpu_id}')    
                    
        args.patch_len = 200
        if x.shape[-1] % 200 != 0:
            args.seq_len = int(x.shape[-1]//args.patch_len)
            x = x[:, :args.patch_len*args.seq_len]
        x = x.reshape(bsz, ch_num, -1, args.patch_len)

        logit = model(x)

        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y

    @staticmethod
    def load_pretrained_weights(args):

        from timm.models import create_model
        from model.LaBraM import modeling_finetune      # must import

        _ = modeling_finetune.NeuralTransformer

        model = create_model(
            'labram_base_patch200_200',
            pretrained=False,
            num_classes=args.n_class,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            use_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            qkv_bias=args.qkv_bias,
        )

        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        return model

    @staticmethod
    def freeze_part(args, model_clsf):
        if args.tune_a_part:
            for name, param in model_clsf.named_parameters():
                if 'fc_norm.' in name or 'head.' in name or \
                    '.attn.v_bias' in name or '.attn.q_bias' in name:
                    continue
                param.requires_grad = False

        return model_clsf