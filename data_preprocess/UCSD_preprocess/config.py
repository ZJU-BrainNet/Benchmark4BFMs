from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    high_pass_filter: float = 0.01
    notch_filter: float = 60
    quality_factor: float = 30

    patch_secs: float = 1
    subject_num: int = 15
    seq_len: int = 10   # secs
    sfreq: int = 256    # 512 downsampled
    group_num: int = 5

    data_root: str = '.../physio_signal_dataset/UCSD_Parkinson/ds002778-download'
    data_save_dir: str = '.../benchmark/datasets/UCSD'
