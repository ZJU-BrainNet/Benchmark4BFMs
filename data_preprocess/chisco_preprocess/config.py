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
    group_num: int = 5
    normal_ratio: float = 3

    data_root: str = '.../physio_signal_dataset/Chisco/preprocessed_pkl'
    data_save_dir: str = '.../benchmark/datasets/Chisco'
    label_path: str = '.../physio_signal_dataset/Chisco/json/textmaps.json'
    class_path: str = '.../physio_signal_dataset/Chisco/json/classnumber.json'