from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    high_pass_filter: float = 0.01
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 5
    sfreq: int = 500
    seq_len: int = 1650   # 3.3 secs
    group_num: int = 4

    data_root: str = '/data/brainnet/physio_signal_dataset/Chisco'
    data_save_dir: str = '/data/brainnet/benchmark/datasets/Chisco'