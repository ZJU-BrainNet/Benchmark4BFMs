from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    min_duration: int = 60
    high_pass_filter: float = 0.01
    notch_filter: float = 50
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 36
    seq_len: int = 10  # secs
    sfreq: int = 250
    group_num: int = 4

    data_root: str = '/data/brainnet/physio_signal_dataset/EEGMat'
    data_save_dir: str = '/data/brainnet/benchmark/datasets/EEGMat'
