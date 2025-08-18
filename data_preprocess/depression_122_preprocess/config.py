from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    min_duration: int = 60
    high_pass_filter: float = 0.01
    notch_filter: float = 60
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 122
    sfreq: int = 250
    seq_len: int = 10  # secs
    group_num: int = 5

    data_root: str = '/data/brainnet/physio_signal_dataset/depression_rest_EEG/ds003478-download'
    data_save_dir: str = '/data/brainnet/benchmark/datasets/Depression_122'
