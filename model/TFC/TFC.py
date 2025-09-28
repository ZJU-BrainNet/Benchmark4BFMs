import torch
from torch import nn, optim
from argparse import Namespace
from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

from model.TFC.loss import NTXentLoss_poly
from model.model_config import ModelPathArgs

class TFC_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim = 1024
        # args.TSlength_aligned = args.patch_len*args.seq_len
        args.TSlength_aligned = 178
        args.temperature = 0.2
        args.use_cosine_similarity = True
        return args

    @staticmethod
    def clsf_loss_func(args):
        if args.n_class != 2:
            ce_weight = [1.0 for _ in range(args.n_class)]
        else:
            ce_weight = [0.3, 1]
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
                {'params': list(model.parameters()), 'lr': args.model_lr},
                # {'params': list(clsf.parameters()), 'lr': args.clsf_lr},
        ],
            betas=(0.9, 0.99), eps=1e-8, weight_decay=3e-4,
        )

    @staticmethod
    def scheduler(optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.n_class)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred


class TFC_Model(nn.Module):
    def __init__(self, configs):
        super(TFC_Model, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=2, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=2,)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
    

class TFC(nn.Module):
    def __init__(self, args):
        super(TFC, self).__init__()
        self.TFC = self.load_pretrained_weights(args)
        self.cls = target_classifier(args)
        self.freeze_part(self)
        self.loss = NTXentLoss_poly(args.device, args.batch_size, args.temperature,
                                    args.use_cosine_similarity)
        
    def forward(self, data, aug1, data_f, aug1_f):
        h_t, z_t, h_f, z_f = self.TFC(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.TFC(aug1, aug1_f)
        loss_t = self.loss(h_t, h_t_aug)
        loss_f = self.loss(h_f, h_f_aug)
        l_TF = self.loss(z_t, z_f)

        l_1, l_2, l_3 = self.loss(z_t, z_f_aug), self.loss(z_t_aug, z_f), \
                        self.loss(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test."""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        predictions = self.cls(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        
        return loss_t, loss_f, loss_c, l_TF, predictions, fea_concat_flat
    

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        data, labels, aug1, data_f, aug1_f = data_packet
        data, labels = data.float(), labels.long()
        data_f = data_f.float()
        aug1 = aug1.float()
        aug1_f = aug1_f.float()
        
        bsz, ch_num, N = data.shape
        loss_t, loss_f, loss_c, l_TF, predictions, fea_concat_flat = model(data, aug1, data_f, aug1_f)
        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels)

        if args.run_mode != 'test':
            loss_p = loss_func(predictions, labels)
            lam = 0.1
            loss = loss_p + l_TF + lam*(loss_t + loss_f)
            return loss, predictions, labels
        else:
            return predictions, labels
        

    @staticmethod
    def load_pretrained_weights(args):
        pretrained_model_path = ModelPathArgs.TFC_path
        chkpoint = torch.load(pretrained_model_path, map_location=args.device)
        pretrained_dict = chkpoint["model_state_dict"]
        TFC_model = TFC_Model(args)
        TFC_model.load_state_dict(pretrained_dict)

        return TFC_model

    @staticmethod
    def freeze_part(model):
        for param in model.TFC.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = True

        return model