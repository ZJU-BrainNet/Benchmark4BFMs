# /data/share/data/neural_decoding/concept_data/SHi_Hongbo/design2/extract_data
# 600 - 2700, 1000Hz
import os
import numpy as np
import pickle
from data_preprocess.chisco_preprocess.config import PreprocessArgs



def split_data(data, label, num_groups, path, subject):
    save_path = os.path.join(path, f'sub-0{subject}', 'group_data')
    os.makedirs(save_path, exist_ok=True)  # 创建保存目录

    data_per_group = len(data) // num_groups
    
    for i in range(num_groups):
        # 计算当前组的起始和结束索引
        start_idx = i * data_per_group
        end_idx = (i + 1) * data_per_group if i != num_groups - 1 else len(data)
        
        # 分割数据和标签
        group_data = data[start_idx:end_idx]
        group_label = label[start_idx:end_idx]
        
        # 保存为.npy文件
        np.save(os.path.join(save_path, f'group_{i}_data.npy'), group_data)
        np.save(os.path.join(save_path, f'group_{i}_label.npy'), group_label)
        
        

def get_sub_data(root_path, args):
    root_path = os.path.join(args.data_root, 'preprocessed_pkl', f'sub-0{subject}', 'eeg')
    data_list = []
    labels_list = []
    for file in os.listdir(root_path):
        if 'imagine' not in file:
            continue

        file_path = os.path.join(root_path, file)
        
        with open(file_path, 'rb') as f:
            pickles = pickle.load(f)
        
        for trial in pickles:
            input_features = trial['input_features'][0, :122, :args.seq_len] * 1e6
            label_text = trial['text'].strip()
            
            data_list.append(input_features.astype(np.float32))
            labels_list.append(label_text)

    if len(data_list) == 0:
        raise ValueError(f"No data loaded for subject {subject}")
        
    data_array = np.stack(data_list)
    labels_array = np.array(labels_list)
    textmap_path =  os.path.join(root_path, 'json/textmaps.json')
    
    mapped_labels = np.array([textmap_path.get(lbl, -1) for lbl in labels_array], dtype=np.int32)
    
    valid_indices = mapped_labels >= 0
    data_array = data_array[valid_indices]
    mapped_labels = mapped_labels[valid_indices]
    print(f"Final data shape: {data_array.shape}, labels shape: {mapped_labels.shape}")
    
    return data_array, mapped_labels
    

root_path = '/data/share/data/neural_decoding/concept_data/SHi_Hongbo/design2/extract_data'
args = PreprocessArgs()
for subject in range(1, args.subject_num+1):
    datas, labels = get_sub_data(root_path, args)
    split_data(datas, labels, args.group_num, args.data_save_dir, subject)