from dataclasses import dataclass


@dataclass
class PreprocessArgs:

    min_duration: int = 60
    high_pass_filter: float = 0.01
    notch_filter: float = 50
    quality_factor: float = 30
    l_freq: float = 0.3
    h_freq: float = 75

    patch_secs: float = 1
    subject_num: int = 15
    seq_len: int = 4  # secs
    sfreq: int = 200
    group_num: int = 6

    data_root: str = '.../physio_signal_dataset/SEED_IV/eeg_raw_data'
    data_save_dir: str = '.../benchmark/datasets/SEED_IV'
