from dataclasses import dataclass


@dataclass
class ModelArgs:
    d_model: int = 768
    time_n_layers: int = 10
    time_n_heads: int = 16

    channel_n_layers: int = 2
    channel_n_heads: int = 16

    norm_eps: float = 1e-5

    seq_len: int = 60  # 60s
    dropout: float = 0.1

    embedding_out_channel: int = 16
    h: int = 31
    w: int = 8

    mask_ratio: float = 0.5
    spec_t_size: int = 9
    spec_f_size: int = 256 // 4  # max frequency components=256Hz
    attention: str = 'eager'  # 'flash_attention_2', 'eager', 'sdpa'