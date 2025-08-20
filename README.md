# BrainBenchmark
The benchmark of self-supervised brain foundation models on brain electric signals. 


## Table of Contents
- [📋 Overview](#overview)
- [⚙️ Get Started](#start)
    * [🗄️ Data Preprocessing](#dataset)
    * [💻 Finetune and Evaluate](#finetune)
- [🪄 How to Extend](#extend)
    * [📚 Add new dataset](#newdata)
    * [🌟 Add new methods](#newmodel)
- [🎯 Benchmark Table](#result)


<h2 id="overview"> 📋 Overview </h2>

Brain electric signals (e.g., EEG, iEEG) are pivotal for understanding neurological and psychiatric disorders, yet progress in AI-driven analysis has been hindered by three systemic challenges: (1) fragmented datasets with inconsistent preprocessing, (2) task-specific evaluation biases, and (3) lack of cross-domain benchmarking for foundation models. Thus, We introduce a benchmark to address these gaps through dataset diversity and Task-Aware evaluation measures.

<h2 id="start"> ⚙️ Get Started </h2>

Please install the following requirements:
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.2.2
scipy==1.10.1
torch==2.0.1
tqdm==4.64.1
```
**Quick Strart**
```
python pretrained_run.py --run_mode finetune --gpu_id 0 --model LaBraM --dataset MAYO --cv_id 0 --batch_size 128
```


<h3 id="dataset"> 🗄️ Data Preprocessing </h3>

For each dataset you want to run experiments on, the first thing to do is generating a specific set of data on your device. This code provides standardized preprocessing pipelines for multiple widely-used datasets, including: [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), [MAYO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [UCSD](https://openneuro.org/datasets/ds002778/versions/1.0.5), [SleepEDFx](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/), [Depression_122](https://doi.org/10.18112/openneuro.ds003478.v1.1.0), [Schizophrenia_28](https://doi.org/10.18150/repod.0107441), [ADHD_Adult](https://doi.org/10.17632/6k4g25fhzg.1), [ADHD_Child](https://doi.org/10.21227/rzfh-zn36). 

Download datasets to obtain the raw data files for your target data. Then, modify `config.py` to specify:
* `data_root` - Root directory where raw datasets are stored.
* `data_save_dir` - Output directory for preprocessed data.


<h3 id="finetune"> 💻 Finetune and Evaluate </h3>

This benchmark support the training and evaluating of the models with a pretrained checkpoint. You should update the path of checkpoints in `model/model_config.py`. 

### Pipeline

To load a checkpoint and train or evaluate from the checkpoint, please run the `pretrained_run.py`. For the `--run_mode` parameter, you can choose one from these strings:

- `finetune`: load the checkpoint, and begin finetune from the checkpoint.
- `test`: evaluate the model with this checkpoint.


### Note

**Fair comparison.** If you want to evaluate a result and make a direct performance comparison with other models on the same dataset, the following arguments about input data must be set according to a unified setting. These arguments includes `dataset`, `seq_len`, `patch_len`. 

**Channel fusion.** Since some methods can only handle single-channel data, an *inter-channel CNN* is used to aggregate the representations across all channels to obtain a "subject-level" representation. The model architecture of this CNN can be set with the following arguments: `cnn_in_channels`, `cnn_kernel_size`. 

**Loading checkpoints.** If you need to load ckpt (continue training from the last breakpoint), please add the `load_ckpt_path` argument (`None` if train from scratch). The path to save model checkpoints can also be set with the `save_ckpt_path` argument. 


<h2 id='extend'>🪄 How to Extend</h2>

<h3 id='newdata'>📚 Add new dataset</h3>

1. Split all the subjects in the new dataset into several groups (4-6 groups are recommended). Each group of data should be generated as a signle file named like `group_0_data.npy`, ..., `group_5_data.npy`. In each file, the shape of the numpy array is: `(seq_num, ch_num, seq_len * patch_len)` 

   The cooresponding label files should be named in similar format: `group_0_label.npy`, ..., `group_5_label.npy`. 

2. Then, add a new element in the `data_info_dict` from  `data_process/data_info.py`. Taking MAYO as an example: 

   ```python
   'MAYO': {'data_path': '.../MAYO/group_data',
        'group_num': 6,
        'split': [3, 1, 2],
        'various_ch_num': False,
        'n_class': 2,
        'sfreq': 1000,
        'channel': 1,
        'seq_len': 3,
        'downstream': 'disorder',
    },
   ```

   - `split`: how to split the `group_num` groups, as training/validation/testing respectively.
   - `various_ch_num`: whether or not the channel number may varies between different data files in this dataset.
   - `n_class`: the task performed on this dataset is a n-class classification task.
   - `sfreq`: the sampling rate of brain signals. 
   - `downstream`: the datasets correspond to specific downstream tasks, with certain models (e.g., NeuroLM) serving as prompts for guided inference.

3. To extend preprocessing to new datasets (denoted as `NAME`), create a dedicated directory `data_preprocess/NAME_preprocess/` containing: 
(1) a configuration file (`config.py`) specifying key parameters including target sampling rate (`sfreq`), notch filtering (`notch_filter`), and high-pass filtering (`high_pass_filter`);
(2) an implementation script (`preprocess.py`) that invokes the core `_segment_data()` utility function to perform signal filtering, segmentation, and sample rate normalization. This modular design ensures consistent preprocessing across datasets while permitting dataset-specific parameterization.

4. In the `utils/meta_info.py`, assume the new dataset name is `NAME`, 

   - Add a line `'NAME': default_get_data,` to the dictionary `get_data_dict`. 
   - Add a line `'NAME': BinaryClassMetrics, ` (if `n_class==2`) or `'NAME': MultiClassMetrics, ` (if `n_class>=3`)  to the dictionary `metrics_dict`. 


<h3 id='newmodel'>🌟 Add new methods</h3>

Assume that the method name is `NAME`, 

1. Make a new directory `model/NAME/`. 

2. Make a new file named `NAME.py` here, and write two classes in this file: `NAME_Trainer` and `NAME`. 

   The class `NAME_Trainer` must includes these functions as members:

   - `def set_config(args: Namespace)` : A static method that sets all of the method's unique parameters as input arguments, such that any user can set these arguments. Taking TF-C model as an example:  

   ```python
       @staticmethod
       def set_config(args: Namespace):
         args.Context_Cont_temperature = 0.2
         args.Context_Cont_use_cosine_similarity = True
         args.augmentation_max_seg = 12
         args.augmentation_jitter_ratio = 2
         args.augmentation_jitter_scale_ratio = 1.5
         return args
   ```

   - `def clsf_loss_func(args)` : A static method that returns the loss function used by this method. Taking TF-C model as an example:  

   ```python
       @staticmethod
       def clsf_loss_func(args):
           return nn.CrossEntropyLoss()
   ```

   - `def optimizer(args, model, clsf) ` : A static method that returns the optimizer used by this method. Taking TF-C model as an example:

   ```python
       @staticmethod
       def optimizer(args, model, clsf):
           return torch.optim.AdamW([
               {'params': list(model.parameters()), 'lr': args.lr},
               {'params': list(clsf.parameters()), 'lr': args.lr},
           ],
               betas=(0.9, 0.95), eps=1e-5,
           )
   ```

   The class `NAME` must includes these functions as members:

   - `def forward_propagate(args, data_packet, model, clsf, loss_func=None)` : based on the data batch `data_packet` (this is determined by the `NAME_dataset` you write later), write the code for model forward propagation and loss calculation. If the code is different between the self-/unsupervision phase and fine-tuning phase, you can use the argument `args.run_mode`  to branch. Taking TF-C model as an example:

   ```python
       @staticmethod
       def forward_propagate(args, data_packet, model, clsf, loss_func=None):
           # x: (bsz, ch_num, seq_len, patch_len)
           # y: (bsz, )
           device = next(model.parameters()).device

           x, y, aug1_x, f, aug1_f = data_packet
           # code to perform fine-tuning

           if args.train_mode == "finetune":
               loss = loss_func(logit, y)
               return loss, logit, y
           elif args.train_mode == "test":
                return logit, y
           else:
               raise NotImplementedError(f'Undefined training mode {args.train_mode}')
   ```

3. Then add any other files about your model in the directory `model/NAME/` to implement the method. 

4. For some methods, they require unique data process (like calculating the spectral density and so on), therefore this benchmark supports to add any new Dataset class for a new method. 

   Make a new file `datasets/NAME_dataset.py`, and write your dataset class `NAME_Dataset` here. Please make sure that the data tuple returned in the `__getitem__` function matches what you receive in the `forward_propagate` function. Make sure that your class contains the following basic member functions: `__len__`, `get_data_loader`. Taking `Braint1_Dataset` as an example: 

   ```python
   class Brant1_Dataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, seq_len, patch_len)
        # y: (seq_num, )
        self.seq_num, self.ch_num, N = x.shape
        x = _std_data_segment(x)    # time level normalization
        x = x.reshape(self.seq_num, self.ch_num, -1, args.patch_len)

        self.x = x
        self.y = y

        self.power = self.compute_power(x, fs=256)

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)

    def __getitem__(self, index):
        return self.x    [index, :, :, :], \
               self.power[index, :, :, :], \
               self.y    [index,]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)

    @staticmethod
    def compute_power(x, fs):
        f, Pxx_den = signal.periodogram(x, fs)

        f_thres = [4, 8, 13, 30, 50, 70, 90, 110, 128]
        poses = []
        for fi in range(len(f_thres) - 1):
            cond1_pos = np.where(f_thres[fi] < f)[0]
            cond2_pos = np.where(f_thres[fi + 1] >= f)[0]
            poses.append(np.intersect1d(cond1_pos, cond2_pos))

        ori_shape = Pxx_den.shape[:-1]
        Pxx_den = Pxx_den.reshape(-1, len(f))
        band_sum = [np.sum(Pxx_den[:, band_pos], axis=-1) + 1 for band_pos in poses]
        band_sum = [np.log10(_band_sum)[:, np.newaxis] for _band_sum in band_sum]
        band_sum = np.concatenate(band_sum, axis=-1)
        ori_shape += (8,)
        band_sum = band_sum.reshape(ori_shape)

        return band_sum
    ```

   If your model is so trivial that it just need raw data `x` and labels `y` as the input of the `forward_propagate` function, you can directly use the class `DefaultDataset` in the `default_dataset.py`. Thus there's no need to write your own dataset class!

5. In the `utils/meta_info.py`, 

   - Add a line `'NAME': NAME_Trainer,` to the dictionary `trainer_dict`, and import the class `NAME_Trainer` here. 
   - Add a line `'NAME': NAME,` to the dictionary `model_dict`, and import the class `NAME` here. 
   - Add a line `'NAME': NAME_Dataset, ` to the dictionary `dataset_class_dict`, and import the model class `NAME_Dataset` here. 

By the steps above, a new method can be added to the benchmark. 


<h2 id="result"> 🎯 Benchmark Table </h2>

### Model

| Mode Name | paper | code |
| ---------- | ---------- | ---------- |
| BIOT | Biot: Biosignal transformer for cross-data learning in the wild | [BIOT](https://github.com/ycq091044/BIOT)|
| BrainBERT | Brainbert: Self-supervised representation learning for intracranial recordings | [Brainbert](https://github.com/czlwang/BrainBERT) |
| Brant1 | Brant: Foundation model for intracranial neural signal | [Brant](https://zju-brainnet.github.io/Brant.github.io/)
| BrainWave | BrainWave: A Brain Signal Foundation Model for Clinical Applications | - |
| Bendr | Bendr: using transformers and a contrastive self-supervised learning task to learn from massive amounts of eeg data | [Bendr](https://github.com/SPOClab-ca/BENDR) |
| LaBraM | Large brain model for learning generic representations with tremendous EEG data in BCI | [LaBraM](https://github.com/935963004/LaBraM) |
| SppEEGNet | Spp-eegnet: An input-agnostic self-supervised eeg representation model for inter-dataset transfer learning | [Spp-eegnet](https://github.com/imics-lab/eeg-transfer-learning) |
| Mbrain | Mbrain: A multi-channel self-supervised learning framework for brain signals | [Mbrain](https://github.com/ZJU-BrainNet/MBrain) |
| BFM | General-Purpose Brain Foundation Models for Time-Series Neuroimaging Data | [BFM](https://mohammadjavadd.github.io/old_homepage/links/bfm2024) |
| CBraMod | CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding | [CBraMod](https://github.com/wjq-learning/CBraMod) |
| Neuro-GPT | Neuro-GPT: developing/towards a foundation model for EEG | [Neuro-GPT](https://github.com/wenhui0206/NeuroGPT) |
| EEGPT | Eegpt: Pretrained transformer for universal and reliable representation of eeg signals | [Eegpt](https://github.com/BINE022/EEGPT) |
| NeuroLM | NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals | [NeuroLM](https://github.com/935963004/NeuroLM) |

### Dataset
The benchmark contains 12 public datasets. 
[CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), [MAYO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [UCSD-ON & UCSD-OFF](https://openneuro.org/datasets/ds002778/versions/1.0.5), [SleepEDFx](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/), [Depression_122_BDI & Depression_122_STAI](https://doi.org/10.18112/openneuro.ds003478.v1.1.0), [Schizophrenia_28](https://doi.org/10.18150/repod.0107441), [ADHD_Adult](https://doi.org/10.17632/6k4g25fhzg.1), [ADHD_Child](https://doi.org/10.21227/rzfh-zn36). 

## Benchmark

| Mode Name | Dataset | Acc | Prec | Rec | F2 | AUCROC | AUPRC |
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|   Brant1  |  CHB_MIT | $68.999 \pm 1.229$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $31.194 \pm 2.172$ | $49.295 \pm 2.651$ |
|   Brant1 | FNUSA | $82.494 \pm 5.399$ | $71.349 \pm 14.661$ | $77.705 \pm 18.389$ | $74.541 \pm 12.918$ | $84.299 \pm 3.324$ | $88.767 \pm 7.377$ |
|   Brant1 | MAYO | $89.653 \pm 3.462$ | $64.193 \pm 22.641$ | $67.176 \pm 15.318$ | $64.809 \pm 14.333$ | $68.330 \pm 17.399$ | $92.031 \pm 4.881$ |
|   Brant1 | UCSD_OFF | $42.675 \pm 25.547$ | $14.512 \pm 13.596$ | $60.000 \pm 54.772$ | $36.672 \pm 33.734$ | $33.282 \pm 7.262$ | $55.089 \pm 7.001$ |
|   Brant1 | UCSD_ON | $44.614 \pm 25.628$ | $18.676 \pm 11.409$ | $57.513 \pm 51.811$ | $36.314 \pm 32.043$ | $30.583 \pm 9.772$ | $52.476 \pm 9.549$ |
|   Brant1 | Depression_122_BDI | $96.382 \pm 0.525$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $4.352 \pm 1.300$ | $52.101 \pm 9.800$ |
|   Brant1 | Schizophrenia_28 | $78.422 \pm 4.739$ | $78.482 \pm 4.739$ | $100.000 \pm 0.000$ | $94.749 \pm 1.422$ | $78.945 \pm 8.080$ | $50.950 \pm 8.744$ |
|   Brant1 | ADHD_Adult | $28.327 \pm 0.232$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $67.702 \pm 6.947$ | $43.703 \pm 14.877$ |
|   Brant1 | ADHD_Child | $28.327 \pm 0.232$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $67.702 \pm 6.947$ | $43.703 \pm 14.877$ | 
|   BrainBERT | CHB_MIT | $25.134 \pm 1.184$ | $25.137 \pm 1.187$ | $99.947 \pm 0.119$ | $62.626 \pm 1.461$ | $26.027 \pm 2.105$ | $51.039 \pm 1.739$ |
|   BrainBERT | FNUSA | $43.468 \pm 6.034$ | $33.250 \pm 9.114$ | $92.706 \pm 8.490$ | $66.744 \pm 4.919$ | $66.168 \pm 14.715$ | $78.036 \pm 13.829$ |
|   BrainBERT | MAYO | $78.600 \pm 11.048$ | $47.727 \pm 28.636$ | $53.177 \pm 30.139$ | $40.727 \pm 19.759$ | $46.907 \pm 21.190$ | $85.777 \pm 6.248$ |
|   BrainBERT | UCSD_OFF | $50.675 \pm 6.459$ | $50.551 \pm 6.739$ | $94.913 \pm 5.360$ | $80.289 \pm 2.945$ | $52.517 \pm 5.452$ | $53.572 \pm 10.102$ |
|   BrainBERT | UCSD_ON | $53.834 \pm 7.886$ | $52.012 \pm 4.054$ | $85.516 \pm 24.208$ | $74.688 \pm 16.917$ | $46.454 \pm 8.061$ | $44.760 \pm 19.333$ |
|   BrainBERT | Depression_122_BDI | $86.789 \pm 11.298$ | $26.519 \pm 41.407$ | $18.139 \pm 24.329$ | $10.300 \pm 13.488$ | $6.845 \pm 2.532$ | $55.295 \pm 9.558$ |
|   BrainBERT | Schizophrenia_28 | $69.268 \pm 18.416$ | $77.410 \pm 3.958$ | $86.216 \pm 30.822$ | $82.592 \pm 26.452$ | $79.506 \pm 7.422$ | $52.683 \pm 11.049$ |
|   BrainBERT | ADHD_Adult | $82.568 \pm 4.425$ | $93.010 \pm 2.928$ | $81.964 \pm 7.028$ | $83.856 \pm 5.829$ | $96.038 \pm 1.247$ | $90.964 \pm 2.794$ |
|   BrainBERT | ADHD_Child | $00.222 \pm 2.100$ | $80.292 \pm 2.100$ | $100.000 \pm 0.000$ | $55.311 \pm 0.592$ | $33.101 \pm 1.225$ | $57.154 \pm 2.318$ |
|   Bendr | CHB_MIT | $51.631 \pm 6.786$ | $26.417 \pm 2.102$ | $51.656 \pm 9.539$ | $42.983 \pm 4.990$ | $26.195 \pm 1.121$ | $52.364 \pm 2.069$ |
|   Bendr | FNUSA | $49.927 \pm 10.692$ | $35.268 \pm 10.997$ | $79.843 \pm 7.170$ | $61.999 \pm 6.770$ | $49.716 \pm 10.618$ | $66.563 \pm 7.106$ |
|   Bendr | MAYO | $79.007 \pm 7.022$ | $25.329 \pm 10.607$ | $19.047 \pm 15.835$ | $17.302 \pm 11.807$ | $21.584 \pm 8.028$ | $61.962 \pm 3.291$ |
|   Bendr | UCSD_OFF | $55.919 \pm 5.848$ | $60.567 \pm 5.585$ | $40.291 \pm 20.915$ | $41.951 \pm 18.942$ | $57.551 \pm 5.214$ | $57.315 \pm 5.374$ |
|   Bendr | UCSD_ON | $50.346 \pm 4.976$ | $52.280 \pm 6.289$ | $51.534 \pm 33.955$ | $47.842 \pm 25.765$ | $50.916 \pm 5.885$ | $49.684 \pm 3.982$ |
|   Bendr | Depression_122_BDI | $74.618 \pm 17.798$ | $5.477 \pm 0.990$ | $25.199 \pm 21.477$ | $12.745 \pm 6.468$ | $6.014 \pm 1.233$ | $50.814 \pm 2.691$ |
|   Bendr | Schizophrenia_28 | $72.756 \pm 3.544$ | $78.726 \pm 4.834$ | $89.573 \pm 1.481$ | $87.111 \pm 1.284$ | $79.393 \pm 5.244$ | $51.620 \pm 1.820$ |
|   Bendr | ADHD_Adult | $59.388 \pm 4.044$ | $72.459 \pm 1.047$ | $69.818 \pm 7.125$ | $70.255 \pm 6.012$ | $71.938 \pm 1.409$ | $50.295 \pm 2.269$ |
|   Bendr | ADHD_Child | $68.587 \pm 3.060$ | $80.453 \pm 2.038$ | $80.446 \pm 4.856$ | $80.400 \pm 3.856$ | $80.082 \pm 1.476$ | $50.201 \pm 1.556$ |
|   BFM | CHB_MIT | $52.335 \pm 13.896$ | $29.527 \pm 3.787$ | $37.249 \pm 34.158$ | $31.442 \pm 21.321$ | $29.858 \pm 2.477$ | $48.237 \pm 5.267$ |
|   BFM | FNUSA | $61.043 \pm 17.556$ | $49.080 \pm 26.049$ | $80.612 \pm 15.926$ | $65.833 \pm 15.209$ | $61.954 \pm 23.867$ | $76.783 \pm 14.347$ |
|   BFM | MAYO | $77.405 \pm 4.418$ | $33.801 \pm 16.642$ | $70.158 \pm 13.441$ | $55.074 \pm 15.400$ | $41.368 \pm 20.513$ | $80.918 \pm 7.292$ |
|   BFM | UCSD_OFF | $27.222 \pm 5.203$ | $25.369 \pm 4.583$ | $98.968 \pm 2.015$ | $62.177 \pm 6.003$ | $29.507 \pm 6.624$ | $56.976 \pm 6.615$ |
|   BFM | UCSD_ON | $28.792 \pm 5.337$ | $23.428 \pm 4.927$ | $83.833 \pm 14.142$ | $55.108 \pm 9.724$ | $21.605 \pm 6.473$ | $43.247 \pm 10.987$ |
|   BFM | Depression_122_BDI | $11.945 \pm 7.002$ | $3.538 \pm 0.456$ | $89.815 \pm 7.039$ | $15.227 \pm 1.567$ | $3.695 \pm 1.384$ | $48.770 \pm 10.083$ |
|   BFM | Schizophrenia_28 | $78.342 \pm 5.828$ | $79.274 \pm 5.440$ | $97.963 \pm 2.004$ | $93.500 \pm 2.575$ | $87.114 \pm 10.376$ | $69.243 \pm 14.832$ |
|   BFM | ADHD_Adult | $82.572 \pm 4.192$ | $85.054 \pm 2.052$ | $91.181 \pm 5.045$ | $89.861 \pm 4.194$ | $92.827 \pm 2.788$ | $87.691 \pm 4.555$ |
|   BFM | ADHD_Child | $79.759 \pm 2.219$ | $80.348 \pm 2.106$ | $98.860 \pm 1.019$ | $94.496 \pm 0.995$ | $86.206 \pm 2.770$ | $64.070 \pm 5.806$ |
|   BIOT | CHB_MIT | $44.360 \pm 20.358$ | $28.749 \pm 12.097$ | $56.874 \pm 37.283$ | $39.981 \pm 22.258$ | $25.774 \pm 3.323$ | $48.502 \pm 1.937$ |
|   BIOT | FNUSA | $72.337 \pm 11.100$ | $54.837 \pm 17.780$ | $80.007 \pm 11.667$ | $71.261 \pm 9.991$ | $74.298 \pm 9.588$ | $83.715 \pm 8.705$ |
|   BIOT | MAYO | $87.248 \pm 4.466$ | $54.517 \pm 17.911$ | $68.318 \pm 12.439$ | $64.025 \pm 13.019$ | $64.117 \pm 16.003$ | $88.772 \pm 5.623$ |
|   BIOT | UCSD_OFF | $49.669 \pm 5.351$ | $23.976 \pm 27.547$ | $32.890 \pm 43.148$ | $30.196 \pm 38.982$ | $48.268 \pm 11.870$ | $44.112 \pm 12.028$ |
|   BIOT | UCSD_ON | $52.084 \pm 2.614$ | $51.166 \pm 4.032$ | $97.786 \pm 2.797$ | $82.664 \pm 3.647$ | $49.986 \pm 17.772$ | $46.118 \pm 18.376$ |
|   BIOT | Depression_122_BDI | $86.349 \pm 12.761$ | $2.755 \pm 2.653$ | $8.576 \pm 12.325$ | $5.151 \pm 7.094$ | $5.400 \pm 1.627$ | $47.242 \pm 10.075$ |
|   BIOT | Schizophrenia_28 | $78.423 \pm 4.817$ | $78.532 \pm 4.824$ | $99.805 \pm 0.225$ | $94.622 \pm 1.461$ | $86.558 \pm 5.066$ | $64.258 \pm 8.115$ |
|   BIOT | ADHD_Adult | $82.293 \pm 3.516$ | $90.779 \pm 3.444$ | $83.892 \pm 2.964$ | $85.164 \pm 2.662$ | $93.693 \pm 2.937$ | $88.470 \pm 4.830$ |
|   BIOT | ADHD_Child | $67.871 \pm 26.562$ | $75.629 \pm 11.160$ | $79.197 \pm 43.391$ | $75.879 \pm 41.291$ | $80.180 \pm 5.331$ | $51.721 \pm 8.917$ |
|   BrainWave | CHB_MIT | $42.582 \pm 20.153$ | $22.318 \pm 4.680$ | $58.475 \pm 42.322$ | $40.303 \pm 24.384$ | $24.427 \pm 2.509$ | $47.858 \pm 5.234$ |
|   BrainWave | FNUSA | $83.683 \pm 8.767$ | $70.535 \pm 19.931$ | $82.683 \pm 8.378$ | $78.942 \pm 10.214$ | $84.376 \pm 8.470$ |
|   BrainWave | MAYO | $92.054 \pm 1.601$ | $69.018 \pm 12.348$ | $74.172 \pm 27.978$ | $71.175 \pm 22.818$ | $74.155 \pm 21.473$ | $87.636 \pm 14.051$ |
|   BrainWave | UCSD_OFF | $53.026 \pm 9.537$ | $52.226 \pm 8.246$ | $98.742 \pm 1.283$ | $83.433 \pm 4.866$ | $58.452 \pm 9.944$ | $57.977 \pm 5.969$ |
|   BrainWave | UCSD_ON | $48.863 \pm 5.221$ | $49.253 \pm 1.844$ | $92.931 \pm 11.076$ | $78.823 \pm 7.449$ | $55.903 \pm 14.659$ | $48.473 \pm 24.723$ |
|   BrainWave | Depression_122_BDI | $83.200 \pm 13.486$ | $1.981 \pm 1.893$ | $7.676 \pm 10.029$ | $4.392 \pm 5.596$ | $5.917 \pm 2.596$ | $47.548 \pm 12.095$ |
|   BrainWave | Schizophrenia_28 | $81.456 \pm 8.498$ | $88.025 \pm 8.336$ | $84.093 \pm 15.047$ | $84.573 \pm 12.935$ | $94.128 \pm 7.437$ | $85.891 \pm 13.372$ |
|   BrainWave | ADHD_Adult | $81.934 \pm 5.575$ | $75.181 \pm 7.904$ | $93.220 \pm 4.710$ | $88.793 \pm 4.061$ | $91.482 \pm 4.892$ | $92.867 \pm 3.749$ |
|   BrainWave | ADHD_Child | $61.420 \pm 6.861$ | $59.408 \pm 5.675$ | $95.840 \pm 3.920$ | $85.280 \pm 4.105$ | $75.946 \pm 6.974$ | $74.010 \pm 7.246$ |
|   CBraMod | CHB_MIT | $36.329 \pm 16.015$ | $18.160 \pm 8.753$ | $57.601 \pm 33.257$ | $38.629 \pm 20.706$ | $19.690 \pm 9.798$ | $39.259 \pm 19.614$ |
|   CBraMod | FNUSA | $61.814 \pm 15.901$ | $47.837 \pm 20.138$ | $64.739 \pm 29.181$ | $55.573 \pm 16.475$ | $64.153 \pm 15.074$ | $73.645 \pm 8.679$ |
|   CBraMod | MAYO | $86.319 \pm 5.323$ | $53.417 \pm 25.676$ | $21.982 \pm 9.799$ | $23.913 \pm 10.219$ | $34.928 \pm 20.792$ | $69.673 \pm 10.219$ |
|   CBraMod | UCSD_OFF | $50.298 \pm 4.699$ | $50.430 \pm 6.652$ | $80.093 \pm 17.491$ | $70.644 \pm 10.874$ | $54.992 \pm 9.559$ | $47.282 \pm 6.531$ |
|   CBraMod | UCSD_ON | $46.119 \pm 2.758$ | $47.088 \pm 3.139$ | $62.372 \pm 16.708$ | $57.960 \pm 12.322$ | $48.614 \pm 5.722$ | $40.589 \pm 4.100$ |
|   CBraMod | Depression_122_BDI | $39.244 \pm 4.220$ | $5.237 \pm 0.327$ | $63.879 \pm 4.008$ | $19.705 \pm 1.081$ | $16.304 \pm 0.772$ | $50.851 \pm 2.773$ |
|   CBraMod | Schizophrenia_28 | $66.480 \pm 5.082$ | $78.529 \pm 4.555$ | $79.951 \pm 11.531$ | $79.278 \pm 8.639$ | $80.586 \pm 4.886$ | $50.862 \pm 3.088$ |
|   CBraMod | ADHD_Adult | $66.526 \pm 14.488$ | $76.144 \pm 4.597$ | $79.120 \pm 30.390$ | $76.604 \pm 26.632$ | $78.718 \pm 10.689$ | $63.626 \pm 16.222$ |
|   CBraMod | ADHD_Child | $68.550 \pm 15.259$ | $80.421 \pm 2.414$ | $79.822 \pm 23.623$ | $79.099 \pm 19.615$ | $81.810 \pm 1.787$ | $50.493 \pm 2.961$ |
|   EEGPT | CHB_MIT | $69.136 \pm 1.705$ | $48.841 \pm 41.768$ | $0.464 \pm 0.270$ | $0.578 \pm 0.336$ | $33.008 \pm 3.105$ | $52.363 \pm 4.113$ |
|   EEGPT | FNUSA | $67.453 \pm 17.954$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $37.216 \pm 13.848$ | $50.313 \pm 0.255$ |
|   EEGPT | MAYO | $86.423 \pm 7.073$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $13.597 \pm 6.898$ | $50.299 \pm 0.636$ |
|   EEGPT | UCSD_OFF | $53.517 \pm 7.475$ | $50.172 \pm 9.031$ | $56.558 \pm 26.262$ | $54.454 \pm 22.025$ | $59.062 \pm 13.164$ | $56.959 \pm 9.185$ |
|   EEGPT | UCSD_ON | $48.498 \pm 19.062$ | $45.053 \pm 21.284$ | $40.401 \pm 28.058$ | $40.626 \pm 26.271$ | $51.688 \pm 20.835$ | $45.827 \pm 26.876$ |
|   EEGPT | Depression_122_BDI | $90.748 \pm 3.569$ | $1.730 \pm 3.868$ | $2.675 \pm 5.982$ | $2.412 \pm 5.393$ | $4.504 \pm 1.673$ | $39.679 \pm 18.118$ |
|   EEGPT | Schizophrenia_28 | $60.212 \pm 10.764$ | $77.793 \pm 11.045$ | $69.104 \pm 8.725$ | $70.533 \pm 8.331$ | $74.023 \pm 13.514$ | $42.969 \pm 22.512$ |
|   EEGPT | ADHD_Adult | $85.252 \pm 3.375$ | $95.977 \pm 2.213$ | $82.996 \pm 5.326$ | $85.244 \pm 4.386$ | $97.619 \pm 0.860$ | $94.457 \pm 1.677$ |
|   EEGPT | ADHD_Child | $68.134 \pm 4.539$ | $85.251 \pm 3.635$ | $72.920 \pm 5.153$ | $75.047 \pm 4.527$ | $85.517 \pm 5.104$ | $63.765 \pm 10.306$ |
|   LaBraM | CHB_MIT | $69.265 \pm 1.342$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $32.529 \pm 5.208$ | $50.932 \pm 5.077$ |
|   LaBraM | FNUSA | $72.970 \pm 8.614$ | $73.857 \pm 10.034$ | $10.962 \pm 8.640$ | $12.964 \pm 9.770$ | $47.457 \pm 11.666$ | $64.589 \pm 7.993$ |
|   LaBraM | MAYO | $85.669 \pm 6.396$ | $36.918 \pm 33.708$ | $3.459 \pm 4.275$ | $4.025 \pm 5.024$ | $27.600 \pm 17.894$ | $69.240 \pm 9.145$ |
|   LaBraM | UCSD_OFF | $48.465 \pm 4.568$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $58.990 \pm 6.011$ | $58.391 \pm 7.638$ |
|   LaBraM | UCSD_ON | $48.872 \pm 5.416$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $45.976 \pm 14.153$ | $34.314 \pm 27.847$ |
|   LaBraM | Depression_122_BDI | $94.904 \pm 0.339$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $4.641 \pm 1.177$ | $44.018 \pm 15.423$ |
|   LaBraM | Schizophrenia_28 | $55.519 \pm 16.460$ | $75.990 \pm 11.196$ | $62.524 \pm 22.373$ | $63.958 \pm 20.656$ | $72.179 \pm 12.429$ | $35.482 \pm 19.001$ |
|   LaBraM | ADHD_Adult | $87.939 \pm 3.137$ | $94.335 \pm 2.755$ | $88.516 \pm 2.472$ | $89.614 \pm 2.322$ | $97.192 \pm 1.018$ | $93.851 \pm 1.791$ |
|   LaBraM | ADHD_Child | $72.066 \pm 7.409$ | $85.340 \pm 3.819$ | $78.573 \pm 7.718$ | $79.778 \pm 6.846$ | $87.516 \pm 6.272$ | $67.369 \pm 12.239$ |
|   Mbrain | CHB_MIT | $56.983 \pm 8.124$ | $24.220 \pm 4.164$ | $34.203 \pm 13.108$ | $30.575 \pm 8.696$ | $25.079 \pm 2.618$ | $48.648 \pm 3.380$ |
|   Mbrain | FNUSA | $78.008 \pm 7.524$ | $60.827 \pm 10.598$ | $74.296 \pm 10.739$ | $70.513 \pm 8.277$ | $75.538 \pm 6.081$ | $84.015 \pm 8.757$ |
|   Mbrain | MAYO | $82.194 \pm 6.581$ | $45.818 \pm 22.764$ | $70.278 \pm 7.202$ | $60.892 \pm 9.474$ | $54.378 \pm 20.413$ | $86.761 \pm 5.213$ |
|   Mbrain | UCSD_OFF | $49.994 \pm 5.659$ | $49.610 \pm 5.365$ | $84.408 \pm 12.999$ | $73.390 \pm 7.821$ | $47.537 \pm 4.606$ | $49.351 \pm 4.889$ |
|   Mbrain | UCSD_ON | $48.526 \pm 4.376$ | $48.823 \pm 3.163$ | $94.030 \pm 10.464$ | $79.183 \pm 7.003$ | $46.611 \pm 2.708$ | $43.934 \pm 9.317$ |
|   Mbrain | Depression_122_BDI | $49.248 \pm 8.635$ | $5.049 \pm 0.735$ | $50.052 \pm 10.290$ | $17.807 \pm 2.108$ | $5.156 \pm 1.230$ | $49.777 \pm 5.511$ |
|   Mbrain | Schizophrenia_28 | $71.805 \pm 4.245$ | $76.503 \pm 4.632$ | $91.618 \pm 8.585$ | $87.898 \pm 5.969$ | $74.744 \pm 2.757$ | $48.295 \pm 6.136$ |
|   Mbrain | ADHD_Adult | $82.968 \pm 4.473$ | $88.102 \pm 3.866$ | $86.253 \pm 6.803$ | $86.515 \pm 5.488$ | $93.033 \pm 3.367$ | $88.399 \pm 5.098$ |
|   Mbrain | ADHD_Child | $73.847 \pm 4.371$ | $75.004 \pm 3.971$ | $96.166 \pm 2.558$ | $90.995 \pm 2.444$ | $78.851 \pm 6.137$ | $61.000 \pm 9.409$ |
|   NeuroGPT | CHB_MIT | $75.141 \pm 0.980$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $25.374 \pm 1.381$ | $50.903 \pm 1.857$ |
|   NeuroGPT | FNUSA | $68.893 \pm 16.387$ | $46.076 \pm 43.316$ | $6.634 \pm 9.615$ | $7.757 \pm 10.991$ | $46.143 \pm 22.508$ | $62.439 \pm 9.422$ |
|   NeuroGPT | MAYO | $82.331 \pm 13.939$ | $4.133 \pm 6.717$ | $4.947 \pm 11.741$ | $4.275 \pm 10.012$ | $12.307 \pm 5.361$ | $48.192 \pm 7.208$ |
|   NeuroGPT | UCSD_OFF | $56.057 \pm 7.509$ | $57.438 \pm 9.569$ | $64.986 \pm 30.644$ | $60.485 \pm 24.259$ | $53.876 \pm 6.968$ | $52.223 \pm 10.380$ |
|   NeuroGPT | UCSD_ON | $46.174 \pm 5.127$ | $38.302 \pm 9.403$ | $38.404 \pm 32.898$ | $36.386 \pm 28.213$ | $44.971 \pm 4.693$ | $48.022 \pm 5.009$ |
|   NeuroGPT | Depression_122_BDI | $94.904 \pm 0.339$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $4.807 \pm 1.002$ | $46.825 \pm 5.958$ |
|   NeuroGPT | Schizophrenia_28 | $60.908 \pm 17.344$ | $75.930 \pm 4.759$ | $74.697 \pm 33.549$ | $72.553 \pm 27.964$ | $75.449 \pm 5.908$ | $48.702 \pm 1.964$ |
|   NeuroGPT | ADHD_Adult | $41.996 \pm 13.876$ | $65.044 \pm 38.538$ | $24.928 \pm 42.628$ | $24.009 \pm 38.319$ | $70.248 \pm 5.935$ | $53.394 \pm 6.663$ |
|   NeuroGPT | ADHD_Child | $53.838 \pm 20.755$ | $59.109 \pm 33.294$ | $58.246 \pm 46.083$ | $56.139 \pm 42.663$ | $75.773 \pm 3.833$ | $55.695 \pm 2.175$ |
|   NeuroLM | CHB_MIT | $61.756 \pm 17.117$ | $6.245 \pm 13.965$ | $20.000 \pm 44.721$ | $13.884 \pm 31.047$ | $38.342 \pm 15.477$ | $50.591 \pm 5.429$ |
|   NeuroLM | FNUSA | $77.904 \pm 13.077$ | $74.900 \pm 18.677$ | $59.021 \pm 17.339$ | $59.748 \pm 13.300$ | $71.542 \pm 13.247$ | $73.339 \pm 7.081$ |
|   NeuroLM | MAYO | $87.432 \pm 5.661$ | $57.202 \pm 33.144$ | $18.472 \pm 14.859$ | $20.466 \pm 15.190$ | $42.355 \pm 16.595$ | $62.063 \pm 7.104$ |
|   NeuroLM | UCSD_OFF | $47.332 \pm 14.430$ | $35.645 \pm 6.510$ | $69.033 \pm 28.733$ | $55.291 \pm 16.311$ | $43.789 \pm 12.300$ | $57.245 \pm 11.006$ |
|   NeuroLM | UCSD_ON | $58.813 \pm 12.414$ | $44.853 \pm 37.621$ | $42.685 \pm 42.992$ | $36.792 \pm 36.073$ | $38.312 \pm 22.556$ | $44.197 \pm 24.652$ |
|   NeuroLM | Depression_122_BDI | $81.545 \pm 15.194$ | $3.605 \pm 3.741$ | $14.859 \pm 14.069$ | $8.661 \pm 8.047$ | $14.818 \pm 21.109$ | $45.344 \pm 11.078$ |
|   NeuroLM | Schizophrenia_28 | $41.312 \pm 21.556$ | $58.566 \pm 33.231$ | $38.154 \pm 41.800$ | $38.293 \pm 38.722$ | $77.553 \pm 6.079$ | $43.864 \pm 7.788$ |
|   NeuroLM | ADHD_Adult | $71.410 \pm 4.654$ | $85.579 \pm 10.915$ | $73.202 \pm 21.852$ | $73.808 \pm 16.208$ | $91.031 \pm 3.422$ | $81.172 \pm 9.209$ |
|   NeuroLM | ADHD_Child | $62.748 \pm 12.354$ | $79.450 \pm 4.708$ | $67.209 \pm 30.285$ | $67.400 \pm 24.830$ | $83.209 \pm 3.763$ | $66.161 \pm 7.176$ |
|   SppEEGNet | CHB_MIT | $49.903 \pm 6.638$ | $23.672 \pm 1.640$ | $46.768 \pm 17.110$ | $38.538 \pm 9.925$ | $33.067 \pm 5.974$ | $49.724 \pm 1.238$ |
|   SppEEGNet | FNUSA | $61.324 \pm 10.373$ | $48.178 \pm 24.169$ | $54.692 \pm 30.772$ | $46.235 \pm 21.896$ | $55.525 \pm 19.007$ | $70.320 \pm 10.814$ |
|   SppEEGNet | MAYO | $68.329 \pm 9.551$ | $25.508 \pm 12.879$ | $50.528 \pm 14.019$ | $38.462 \pm 4.566$ | $30.211 \pm 10.983$ | $65.516 \pm 3.572$ |
|   SppEEGNet | UCSD_OFF | $49.970 \pm 4.448$ | $48.762 \pm 5.007$ | $72.075 \pm 13.445$ | $65.238 \pm 8.709$ | $62.537 \pm 5.135$ | $52.241 \pm 5.380$ |
|   SppEEGNet | UCSD_ON | $51.456 \pm 4.259$ | $50.352 \pm 3.651$ | $65.389 \pm 19.424$ | $60.924 \pm 14.152$ | $64.772 \pm 6.177$ | $53.728 \pm 7.394$ |
|   SppEEGNet | Depression_122_BDI | $50.765 \pm 16.238$ | $5.564 \pm 0.823$ | $54.599 \pm 20.270$ | $19.349 \pm 3.893$ | $20.809 \pm 9.815$ | $53.857 \pm 5.776$ |
|   SppEEGNet | Schizophrenia_28 | $56.455 \pm 3.486$ | $77.634 \pm 6.035$ | $62.847 \pm 4.275$ | $65.242 \pm 3.701$ | $77.958 \pm 6.019$ | $47.444 \pm 4.587$ |
|   SppEEGNet | ADHD_Adult | $66.985 \pm 6.659$ | $80.520 \pm 4.867$ | $71.795 \pm 12.926$ | $73.011 \pm 10.465$ | $81.094 \pm 7.754$ | $66.907 \pm 11.704$ |
|   SppEEGNet | ADHD_Child | $58.937 \pm 4.345$ | $80.328 \pm 2.087$ | $64.924 \pm 6.931$ | $67.403 \pm 5.889$ | $82.693 \pm 2.094$ | $51.602 \pm 5.121$ |


| Mode Name | Dataset | Acc | TopKAcc | TopKAcc | Spec | MF1 | Kappa |
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|   Brant1  |  ISRUC | $21.363 \pm 3.965$ | $71.690 \pm 6.145$ | $80.000 \pm 0.000$ | $80.000 \pm 0.000$ | $7.014 \pm 1.099$ | $0.000 \pm 0.000$ |
|   Brant1  |  SleepEDFx | $78.337 \pm 2.898$ | $98.650 \pm 0.431$ | $93.946 \pm 0.854$ | $93.946 \pm 0.854$ | $71.566 \pm 3.192$ | $69.881 \pm 4.224$ |
|   Brant1  |  Depression_122_STAI | $95.390 \pm 0.625$ | $100.000 \pm 0.000$ | $66.667 \pm 0.000$ | $66.667 \pm 0.000$ | $48.820 \pm 0.163$ | $0.000 \pm 0.000$ |
|   BrainBERT  |  ISRUC | $28.099 \pm 6.324$ | $68.779 \pm 7.816$ | $80.196 \pm 0.461$ | $80.196 \pm 0.461$ | $11.536 \pm 2.024$ | $1.046 \pm 2.442$ |
|   BrainBERT  |  SleepEDFx | $74.673 \pm 2.314$ | $98.370 \pm 0.360$ | $93.150 \pm 0.796$ | $93.150 \pm 0.796$ | $66.980 \pm 4.006$ | $65.767 \pm 4.377$ |
|   BrainBERT  |  Depression_122_STAI | $93.678 \pm 0.705$ | $100.000 \pm 0.000$ | $65.924 \pm 1.662$ | $65.924 \pm 1.662$ | $43.322 \pm 11.155$ | $-0.327 \pm 0.730$ |
|   Bendr |  ISRUC | $23.228 \pm 1.603$ | $64.129 \pm 3.013$ | $80.115 \pm 0.112$ | $80.115 \pm 0.112$ | $18.819 \pm 0.743$ | $0.504 \pm 0.602$ |
|   Bendr |  SleepEDFx | $40.915 \pm 1.789$ | $75.892 \pm 1.039$ | $80.033 \pm 0.209$ | $80.033 \pm 0.209$ | $15.558 \pm 1.185$ | $0.270 \pm 1.176$ |
|   Bendr |  Depression_122_STAI | $66.110 \pm 7.265$ | $100.000 \pm 0.000$ | $68.144 \pm 1.722$ | $68.144 \pm 1.722$ | $29.880 \pm 1.563$ | $1.156 \pm 0.974$ |
|   BFM |  ISRUC | $25.616 \pm 5.806$ | $68.074 \pm 5.098$ | $80.180 \pm 0.384$ | $80.180 \pm 0.384$ | $11.806 \pm 1.303$ | $0.858 \pm 1.861$ |
|   BFM |  SleepEDFx | $69.012 \pm 3.729$ | $96.165 \pm 0.594$ | $90.997 \pm 1.021$ | $90.997 \pm 1.021$ | $57.039 \pm 3.087$ | $55.770 \pm 5.268$ |
|   BFM |  Depression_122_STAI | $93.422 \pm 4.070$ | $100.000 \pm 0.000$ | $66.182 \pm 1.024$ | $66.182 \pm 1.024$ | $48.521 \pm 0.724$ | $-0.974 \pm 1.883$ |
|   BIOT |  ISRUC | $24.543 \pm 1.744$ | $63.139 \pm 4.376$ | $80.006 \pm 0.213$ | $80.006 \pm 0.213$ | $15.675 \pm 2.943$ | $0.166 \pm 1.033$ |
|   BIOT |  SleepEDFx | $67.702 \pm 3.196$ | $95.026 \pm 1.520$ | $91.438 \pm 1.129$ | $91.438 \pm 1.129$ | $59.204 \pm 2.972$ | $57.297 \pm 4.901$ |
|   BIOT |  Depression_122_STAI | $82.159 \pm 14.526$ | $100.000 \pm 0.000$ | $66.780 \pm 2.460$ | $66.780 \pm 2.460$ | $43.837 \pm 7.169$ | $-0.314 \pm 2.731$ |
|   BrainWave |  ISRUC | $25.996 \pm 5.433$ | $66.752 \pm 3.789$ | $79.761 \pm 0.704$ | $79.761 \pm 0.704$ | $8.556 \pm 2.463$ | $-1.047 \pm 3.315$ |
|   BrainWave |  SleepEDFx | $69.686 \pm 5.327$ | $93.737 \pm 6.518$ | $91.381 \pm 3.171$ | $91.381 \pm 3.171$ | $57.856 \pm 11.085$ | $57.055 \pm 14.003$ |
|   BrainWave |  Depression_122_STAI | $91.403 \pm 3.762$ | $100.000 \pm 0.000$ | $67.511 \pm 1.292$ | $67.511 \pm 1.292$ | $46.901 \pm 7.431$ | $2.553 \pm 4.114$ |
|   CBraMod |  ISRUC | $25.924 \pm 5.827$ | $62.082 \pm 6.354$ | $79.993 \pm 0.026$ | $79.993 \pm 0.026$ | $8.741 \pm 4.964$ | $-0.030 \pm 0.124$ |
|   CBraMod |  SleepEDFx | $59.659 \pm 3.911$ | $93.234 \pm 1.853$ | $89.074 \pm 0.693$ | $89.074 \pm 0.693$ | $52.665 \pm 2.342$ | $45.594 \pm 3.058$ |
|   CBraMod |  Depression_122_STAI | $61.998 \pm 9.409$ | $100.000 \pm 0.000$ | $65.739 \pm 0.640$ | $65.739 \pm 0.640$ | $28.381 \pm 2.359$ | $-0.881 \pm 0.339$ |
|   EEGPT |  ISRUC | $26.711 \pm 7.738$ | $70.828 \pm 7.576$ | $80.713 \pm 1.073$ | $80.713 \pm 1.073$ | $12.823 \pm 7.233$ | $3.281 \pm 4.775$ |
|   EEGPT |  SleepEDFx | $43.631 \pm 2.256$ | $76.419 \pm 2.412$ | $80.000 \pm 0.000$ | $80.000 \pm 0.000$ | $12.169 \pm 0.333$ | $0.000 \pm 0.000$ |
|   EEGPT |  Depression_122_STAI | $89.618 \pm 6.805$ | $100.000 \pm 0.000$ | $67.630 \pm 2.527$ | $67.630 \pm 2.527$ | $42.710 \pm 8.343$ | $1.332 \pm 4.189$ |
|   LaBraM  |  ISRUC | $25.046 \pm 5.242$ | $71.594 \pm 6.068$ | $80.255 \pm 0.642$ | $80.255 \pm 0.642$ | $11.050 \pm 3.190$ | $1.225 \pm 3.197$ |
|   LaBraM  |  SleepEDFx | $68.214 \pm 1.497$ | $94.039 \pm 1.162$ | $91.083 \pm 0.840$ | $91.083 \pm 0.840$ | $59.340 \pm 1.441$ | $56.883 \pm 3.911$ |
|   LaBraM  |  Depression_122_STAI | $93.905 \pm 0.702$ | $100.000 \pm 0.000$ | $66.667 \pm 0.000$ | $66.667 \pm 0.000$ | $48.428 \pm 0.187$ | $0.000 \pm 0.000$ |
|   Mbrain |  ISRUC | $26.363 \pm 5.087$ | $69.218 \pm 4.982$ | $80.181 \pm 0.587$ | $80.181 \pm 0.587$ | $12.436 \pm 2.652$ | $0.684 \pm 2.612$ |
|   Mbrain |  SleepEDFx | $71.040 \pm 3.843$ | $97.326 \pm 0.767$ | $92.388 \pm 1.000$ | $92.388 \pm 1.000$ | $64.962 \pm 3.717$ | $61.693 \pm 5.124$ |
|   Mbrain |  Depression_122_STAI | $90.428 \pm 0.615$ | $100.000 \pm 0.000$ | $66.412 \pm 0.906$ | $66.412 \pm 0.906$ | $42.646 \pm 8.576$ | $-0.811 \pm 3.269$ |
|   NeuroGPT |  ISRUC | $20.274 \pm 1.887$ | $57.711 \pm 4.230$ | $79.945 \pm 0.081$ | $79.945 \pm 0.081$ | $8.600 \pm 1.033$ | $-0.282 \pm 0.398$ |
|   NeuroGPT |  SleepEDFx | $43.631 \pm 2.256$ | $75.205 \pm 3.231$ | $80.000 \pm 0.000$ | $80.000 \pm 0.000$ | $12.145 \pm 0.438$ | $0.000 \pm 0.000$ |
|   NeuroGPT |  Depression_122_STAI | $93.678 \pm 0.705$ | $100.000 \pm 0.000$ | $66.667 \pm 0.000$ | $66.667 \pm 0.000$ | $48.367 \pm 0.188$ | $0.000 \pm 0.000$ |
|   NeuroLM  |  ISRUC | $20.130 \pm 1.181$ | $65.539 \pm 6.844$ | $80.053 \pm 0.233$ | $80.053 \pm 0.233$ | $10.008 \pm 1.333$ | $0.287 \pm 1.129$ |
|   NeuroLM  |  SleepEDFx | $36.303 \pm 24.640$ | $82.538 \pm 5.383$ | $83.785 \pm 5.191$ | $83.785 \pm 5.191$ | $26.686 \pm 23.502$ | $18.905 \pm 26.268$ |
|   NeuroLM  |  Depression_122_STAI | $93.298 \pm 1.193$ | $100.000 \pm 0.000$ | $66.591 \pm 0.157$ | $66.591 \pm 0.157$ | $48.435 \pm 0.173$ | $-0.357 \pm 0.737$ |
|   SppEEGNet  |  ISRUC | $21.639 \pm 4.612$ | $64.103 \pm 3.771$ | $80.033 \pm 0.589$ | $80.033 \pm 0.589$ | $18.289 \pm 2.757$ | $0.102 \pm 2.898$ |
|   SppEEGNet  |  SleepEDFx | $37.692 \pm 2.214$ | $76.002 \pm 5.676$ | $82.429 \pm 0.588$ | $82.429 \pm 0.588$ | $24.929 \pm 3.053$ | $12.310 \pm 3.133$ |
|   SppEEGNet  |  Depression_122_STAI | $64.805 \pm 19.181$ | $100.000 \pm 0.000$ | $67.158 \pm 1.993$ | $67.158 \pm 1.993$ | $28.671 \pm 4.824$ | $0.484 \pm 2.199$ |