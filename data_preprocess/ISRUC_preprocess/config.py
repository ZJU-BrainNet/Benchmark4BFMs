from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    high_pass_filter: float = 0.01
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 100
    seq_len: int = 30  # secs
    sfreq: int = 200
    group_num: int = 5

    data_root: str = '/data/brainnet/physio_signal_dataset/ISRUC_dataset/subgroup_1'
    data_save_dir: str = '/data/brainnet/benchmark/datasets/ISRUC'
