import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaModel
from model.BrainWave.model.config import ModelArgs

def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden, bias, dropout):
        super(MLP, self).__init__()
        if hidden:
            hidden_dim = 2*(in_features+out_features) // 3
            self.mlp = nn.Sequential(
                nn.Linear(in_features, hidden_dim, bias=bias),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim, bias=bias),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_features, bias=bias),
            )
        else:
            self.mlp = nn.Linear(in_features, out_features, bias=bias)

        self.apply(_weights_init)

    def forward(self, x):

        return self.mlp(x)

class Embedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Embedding, self).__init__()

        self.t_size = args.spec_t_size
        self.f_size = args.spec_f_size
        self.h = args.h
        self.w = args.w
        self.mask_ratio = args.mask_ratio
        self.embedding_out_channel = args.embedding_out_channel

        self.cls = nn.Parameter(torch.randn(args.d_model))
        self.mask = nn.Parameter(torch.zeros(args.d_model), requires_grad=False)
        self.padding = nn.Parameter(torch.randn(args.embedding_out_channel).unsqueeze(1).unsqueeze(2).unsqueeze(0))

        self.cnn = nn.Conv2d(in_channels=1, out_channels=args.embedding_out_channel, kernel_size=(4, 2), stride=(2, 1))
        self.project_layer = MLP(in_features=args.embedding_out_channel*args.h*args.w, out_features=args.d_model, hidden=False, bias=True, dropout=0.2)


    def forward(self, x, mask):
        bsz, ch_num, seq_len, f_size, t_size = x.shape

        x = x.reshape(-1, 1, f_size, t_size)
        # truncate
        if f_size > self.f_size:
            x = x[:, :, :self.f_size]
        emb = self.cnn(x)
        x = x.reshape(bsz, ch_num, seq_len, -1, t_size)
        a, _, h, w = emb.shape
        # padding
        if h < self.h:
            padding = torch.tile(self.padding, (a, 1, self.h-h, w))
            emb = torch.concat([emb, padding], dim=-2)

        emb = emb.reshape(bsz*ch_num*seq_len, -1)
        emb = self.project_layer(emb)
        emb = emb.reshape(bsz, ch_num*seq_len, -1)

        # dynamic masking
        mask_pos = None
        if mask:
            mask_num = int(self.mask_ratio*ch_num*seq_len)
            mask_pos = np.random.permutation(ch_num*seq_len)[:mask_num]
            emb[:, mask_pos] = self.mask

        # add cls
        emb = emb.reshape(bsz * ch_num, seq_len, -1)
        cls = torch.tile(self.cls, (bsz*ch_num, 1))
        cls = torch.unsqueeze(cls, 1)
        emb = torch.concat([cls, emb], dim=1)
        emb = emb.reshape(bsz, ch_num, seq_len + 1, -1)

        return emb, x, mask_pos


class BrainWaveEncoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super(BrainWaveEncoder, self).__init__()

        self.embedding = Embedding(args=args)
        roberta_config = RobertaConfig(hidden_size=args.d_model,
                                       num_hidden_layers=args.time_n_layers,
                                       num_attention_heads=args.time_n_heads,
                                       intermediate_size=args.d_model * 8 // 3,
                                       max_position_embeddings=args.seq_len+1,
                                       position_embedding_type='absolute')
        self.time_encoder = RobertaModel(roberta_config)
        
        self.channel_attn = nn.MultiheadAttention(args.d_model, args.channel_n_heads, batch_first=True)
        self.q = nn.Linear(args.d_model, args.d_model, bias=False)
        self.k = nn.Linear(args.d_model, args.d_model, bias=False)
        self.v = nn.Linear(args.d_model, args.d_model, bias=False)


    def forward(self, x, mask=True):
        emb, x, mask_pos = self.embedding(x, mask)
        bsz, ch_num, seq_len, _ = emb.shape
        emb = emb.reshape(bsz*ch_num, seq_len, -1)

        t = self.time_encoder(position_ids=torch.tile(torch.arange(seq_len), dims=(bsz*ch_num, 1)).to(x.device), inputs_embeds=emb, use_cache=False, output_attentions=False, output_hidden_states=False).last_hidden_state

        z = torch.swapaxes(t.reshape(bsz, ch_num, seq_len, -1), axis0=1, axis1=2).reshape(bsz*seq_len, ch_num, -1)
        z = self.channel_attn(self.q(z), self.k(z), self.v(z), need_weights=False)[0]
        z = torch.swapaxes(z.reshape(bsz, seq_len, ch_num, -1), axis0=1, axis1=2)

        t = t.reshape(bsz, ch_num, seq_len, -1)

        return t, z, x, mask_pos
    
    
class LinearHead(nn.Module):
    def __init__(self, in_features, class_num, hidden, bias, dropout, mean_pool=False):
        super(LinearHead, self).__init__()
        self.head = MLP(in_features=in_features, out_features=class_num, hidden=hidden, bias=bias, dropout=dropout)
        self.mean_pool = mean_pool

    def forward(self, x):
        if len(x.shape) == 3:
            bsz, _, _ = x.shape
            if self.mean_pool:
                x = torch.mean(x, dim=1)
            else:
                x = x.reshape(bsz, -1)
        x = self.head(x)

        return x
