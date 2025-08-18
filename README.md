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
## Model
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

## Dataset
The benchmark contains 12 public datasets. 
[CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), [MAYO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [UCSD-ON & UCSD-OFF](https://openneuro.org/datasets/ds002778/versions/1.0.5), [SleepEDFx](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/), [Depression_122_BDI & Depression_122_STAI](https://doi.org/10.18112/openneuro.ds003478.v1.1.0), [Schizophrenia_28](https://doi.org/10.18150/repod.0107441), [ADHD_Adult](https://doi.org/10.17632/6k4g25fhzg.1), [ADHD_Child](https://doi.org/10.21227/rzfh-zn36). 

## Benchmark
