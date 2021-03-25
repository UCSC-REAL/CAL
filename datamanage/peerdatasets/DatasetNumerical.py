import os
import sys

import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import random

class DatasetNumerical(Dataset):
    '''
    '''
    # num_classes need to be defined
    def __init__(self, root=None, is_train=True, label_file_path=None, transform=None,
                 selected_idx=None, select_label=True, chosen_classes=None, classes_dist=None, is_mixup=False):
        self.root = os.path.expanduser(root)
        self.is_train = is_train
        self.label_file_path = label_file_path
        self.transform = transform
        self.selected_idx = selected_idx
        self.select_label = select_label
        self.chosen_classes = chosen_classes
        self.classes_dist = classes_dist
        self.is_mixup = is_mixup

        assert hasattr(self, "num_classes")

        self.load_data()
        self.check_dtype()
        self.preprocess()

    def __getitem__(self, index):
        '''
        '''
        feature, label, true_label = self.data[index], self.label[index], self.true_label[index]

        if self.transform is not None:
            if self.is_mixup: # do not use this 
                feature1 = self._transform(feature)
                feature2 = self._transform(feature)
                return feature1, feature2, label, true_label, index
            else:
                feature = self._transform(feature)
                if self.label_file_path is not None:
                    if "noiseID" in self.label_file_path:
                        feature_label = self.feature_label[index]
                        return feature, label, feature_label, true_label, index
                    else:
                        return feature, label, true_label, index
                else:
                    return feature, label, true_label, index
        else:
            return feature, label, true_label, index

    def _transform(self, feature):
        # when pick a batch at a time
        if len(feature.shape) == 4:
            for idx in range(feature.shape[0]):
                feature[idx] = self.transform(feature[idx])
        else:
            feature = self.transform(feature)
        return feature

    def __len__(self):
        return len(self.data)
    
    def get_raw_data(self):
        return self.data, self.label, self.true_label
    
    def load_data(self):
        raise NotImplementedError

    def check_dtype(self):
        r"""making sure the data and label are all torch tensors"""
        # for data
        if not isinstance(self.data, torch.Tensor):
            if isinstance(self.data, np.ndarray):
                self.data = torch.from_numpy(self.data)
            else:
                raise TypeError(f"invalid dtype {type(self.data)}")
        # for label
        if not isinstance(self.label, torch.Tensor):
            if isinstance(self.label, np.ndarray):
                self.label = torch.from_numpy(self.label).long()
            elif isinstance(self.label, list):
                self.label = torch.tensor(self.label, dtype=torch.long)
            else:
                raise TypeError(f"invalid dtype {type(self.data)}")

    def preprocess(self):
        # limit to selected indices; e.g for cotraining & self-supervised learning
        if self.selected_idx is not None:
            # select features & (true) label;
            # NOTE for training part in cotraining we only load part of the label 
            # & need not apply selected_idx; we include flag select_label to control this
            self.data = self.data[ self.selected_idx ]
            self.true_label = self.true_label[ self.selected_idx ]
            if self.select_label:
                self.label = self.label[ self.selected_idx ]
            # check integrity
            if self.data.shape[0] != self.label.shape[0]:
                print(self.data.shape[0], self.label.shape[0])
                raise ValueError("data and label not match")

        # # limit to chosen classes
        # if self.chosen_classes is not None:
        #     #TODO
        #     if self.label_file_path is not None:
        #         print("May cause error when noise labels have members not in the chosen_classes")
        #         raise NotImplementedError
        #     self.num_classes = len(self.chosen_classes)
        #     chosen_class = torch.tensor(self.chosen_classes)
        #     idx = torch.nonzero( self.label.unsqueeze(-1) == chosen_class )
        #     self.data = self.data[ idx[:,0].view(-1) ]
        #     self.label = idx[:,1].view(-1).clone() #convert the chosen indices to 0,1,...
        #     self.true_label = idx[:,1].view(-1).clone()

        # set distribution over diff classes; 
        if self.classes_dist is not None:
            self.set_class_dist()
        else:
            self.get_class_size()
    
    def load_label(self):
        '''
        I adopt .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        #NOTE presently only use for load manual training label
        noise_label = torch.load(self.label_file_path)
        if isinstance(noise_label, dict):
            if "clean_label_train" in noise_label.keys():
                clean_label = noise_label['clean_label_train']
                # assert torch.sum(self.label - clean_label) == 0  # commented for noise identification (NID) since we need to replace labels
            return noise_label['noise_label_train'] # % 10
        else:
            return noise_label  # % 10

    



    def set_class_dist(self):
        selected_idx = []
        class_size, idx_each_class = self.get_class_size()
        ub_list = np.array(self.classes_dist[:-1])
        ub_list = self.classes_dist[-1] * (ub_list/np.sum(ub_list))
        ub_list = ub_list.astype(int)
        for i in range(len(class_size)):
            if ub_list[i]>class_size[i]:
                raise ValueError(f'Too much training data in a class {i}')
            else:
                selected_idx += list(np.array(idx_each_class[i])[:ub_list[i]])
                random.shuffle(selected_idx)
                print(f'accumulated length is {len(selected_idx)}')
        self.data = self.data[selected_idx]
        self.label = self.label[selected_idx]
        self.true_label = self.true_label[selected_idx]
        fname = 'check_data.pkl'
        save_dir = './'
        dict_save = { 'noisy_label': self.label, 'raw_idx':  selected_idx, 'raw_label': self.true_label}
        file_path = os.path.join(save_dir, fname)
        with open(file_path, 'wb') as f:
            pickle.dump(dict_save, f)
        print(f'selected_idx: {selected_idx[:100]}')
        print(f'true_label: {self.true_label[:100]}')
        print(f"data {fname} saved to {save_dir}")
        # exit()

    def get_class_size(self):
        if self.num_classes == 10 and max(self.label) >= 10:
            num_duplicate_label = max(self.label)//10 + 1
            # 1 -> 5 labels
            self.noisy_prior = [[] for _ in range(num_duplicate_label)]
            for clusteridx in range(num_duplicate_label):
                idx_each_class = [[] for i in range(self.num_classes)]
                idx_each_class_noisy = [[] for i in range(self.num_classes)]
                for i in range(self.label.shape[0]):
                    if self.label[i] // 10 == clusteridx:
                        # idx_each_class[self.label[i]].append(i)
                        idx_each_class[self.true_label[i]].append(i)
                        idx_each_class_noisy[self.label[i]%10].append(i)
                class_size = [len(idx_each_class[i]) for i in range(self.num_classes)]
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.num_classes)]
                print(f'cluster {clusteridx}')
                print(f'The original data size in each class is {class_size}')
                print(f'The original data ratio in each class is {np.round((np.array(class_size)/sum(class_size)),2).tolist()}')
                print(f'The noisy data size in each class is {class_size_noisy}')
                print(f'The noisy... data ratio in each class is {np.round((np.array(class_size_noisy)/sum(class_size_noisy)),2).tolist()}\n')
                self.noisy_prior[clusteridx] = (np.array(class_size_noisy)/sum(class_size_noisy)).tolist()

        else:
            idx_each_class = [[] for i in range(self.num_classes)]
            idx_each_class_noisy = [[] for i in range(self.num_classes)]
            for i in range(self.label.shape[0]):
                # idx_each_class[self.label[i]].append(i)
                idx_each_class[self.true_label[i]].append(i)
                idx_each_class_noisy[self.label[i]].append(i)
            for i in range(self.num_classes):
                random.shuffle(idx_each_class[i])
                random.shuffle(idx_each_class_noisy[i])
            class_size = [len(idx_each_class[i]) for i in range(self.num_classes)]
            class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(self.num_classes)]
            print(f'The original data size in each class is {class_size}')
            print(f'The original data ratio in each class is {(np.array(class_size)/sum(class_size)).tolist()}')
            print(f'The noisy data size in each class is {class_size_noisy}')
            self.noisy_prior = (np.array(class_size_noisy)/sum(class_size_noisy)).tolist()
            print(f'The noisy data ratio in each class is {self.noisy_prior}')
        
        return class_size, idx_each_class