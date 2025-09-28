from dataclasses import dataclass


@dataclass
class ModelPathArgs:
    BrainBERT_path: str = '.../benchmark/pretrained_weights/BrainBERT/stft_large_pretrained.pth'
    Brant1_root_path: str = '.../benchmark/pretrained_weights/Brant1/'
    BrainWave_path: str = '.../model_20198955825.pt'
    LaBraM_path: str = '.../benchmark/pretrained_weights/LaBraM/labram-base.pth'
    BIOT_path: str = '.../benchmark/pretrained_weights/BIOT/EEG-six-datasets-18-channels.ckpt'
    Bendr_path: str = '.../benchmark/pretrained_weights/Bendr/contextualizer.pt'
    Bendr_contextualizer_path: str = '.../benchmark/pretrained_weights/Bendr/encoder.pt'
    SppEEGNet_path: str = '.../benchmark/pretrained_weights/SppEEGNet/tuh_all_ckp.pt'
    Mbrain_path: str = '.../MBrain/01_02_06'
    CRT_root_path: str = '.../benchmark/pretrained_weights/crt'
    BFM_path: str = ".../benchmark/pretrained_weights/BFM"
    CBraMod_path: str = '.../benchmark/pretrained_weights/CBraMod/pretrained_weights.pth'
    NeuroGPT_path: str = '.../benchmark/pretrained_weights/NeuroGPT/pytorch_model.bin'
    NeuroLM_path: str = '.../benchmark/pretrained_weights/NeuroLM/NeuroLM-B.pt'
    NeuroLM_token_path: str = '.../benchmark/pretrained_weights/NeuroLM/VQ.pt'
    GPT2_folder_path: str = '/data/pretrain_models/gpt2'
    EEGPT_path: str = '.../benchmark/pretrained_weights/EEGPT/eegpt_mcae_58chs_4s_large4E.ckpt'
    SPaRCNet_path: str = '.../benchmark/pretrained_weights/SPaRCNet/model_1130.pt'
    TFC_path: str = '.../benchmark/pretrained_weights/TFC/ckp_last.pt'