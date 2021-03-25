import os
import os.path
import sys
import pickle
import torch
import numpy as np
from .DatasetNumerical import DatasetNumerical
from utils.functions import check_integrity, download_and_extract_archive


class CIFAR10(DatasetNumerical):
    """
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    num_classes = 10
    
    def __init__(self, root=None, is_train=True, label_file_path=None, transform=None, is_mixup=False,
                selected_idx=None, select_label=True, chosen_classes=None, classes_dist=None, is_download=False):
        self.is_download = is_download
        super(CIFAR10, self).__init__(root=root, is_train=is_train, label_file_path=label_file_path, transform=transform,
                                      selected_idx=selected_idx, select_label=select_label, 
                                      chosen_classes=chosen_classes, classes_dist=classes_dist,is_mixup=is_mixup)

    def load_data(self):
        if self.is_download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.is_train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.label = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.label.extend(entry['labels'])
                else:
                    self.label.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = torch.tensor( self.data, dtype=torch.float ) / 255.
        # self.data = torch.tensor( self.data, dtype=torch.float )  # for generating cores noise
        self.label = torch.tensor( self.label, dtype=torch.long )

        # self.true_label = self.label
        self.true_label = self.label.clone()

        # load label for noise label or co-training cases
        if (self.label_file_path is not None) and self.is_train:
            self.label = self.load_label()
        assert isinstance(self.label, torch.Tensor)

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
    
    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)


class CIFAR10CAL(CIFAR10):
    """
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    num_classes = 10
    
    def __init__(self, root=None, is_train=True, label_file_path=None, transform=None, is_mixup=False,
                selected_idx=None, select_label=True, chosen_classes=None, classes_dist=None, is_download=False, sample_weight_path = None, beta_path = None, ratio_true = 1):
        self.is_download = is_download
        self.sample_weight_path = sample_weight_path
        self.beta_path = beta_path
        self.ratio_true = ratio_true
        super(CIFAR10CAL, self).__init__(root=root, is_train=is_train, label_file_path=label_file_path, transform=transform,
                                      selected_idx=selected_idx, select_label=select_label, 
                                      chosen_classes=chosen_classes, classes_dist=classes_dist,is_mixup=is_mixup, is_download= is_download)

    def load_data(self):
        if self.is_download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.is_train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.label = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.label.extend(entry['labels'])
                else:
                    self.label.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = torch.tensor( self.data, dtype=torch.float ) / 255.
        self.label = torch.tensor( self.label, dtype=torch.long )
        self.true_label = self.label.clone()
        col = len(self.chosen_classes)
        row = len(self.chosen_classes)
        # T_mat = [[ [] for i in range(col)] for i in range(row)]
        # T_mat_true = [[ [] for i in range(col)] for i in range(row)]

        # load label for noise label or co-training cases
        if (self.label_file_path is not None) and self.is_train:
            # self.true_label, self.label, sel_class, sel_idx, self.distilled_label, self.distilled_weight = self.load_label_binary()
            self.true_label, self.label, self.distilled_label, self.distilled_weight = self.load_label_multiple()
            self.true_label_d = self.true_label.clone()
            # self.data = self.data[sel_idx]
            self.T_mat = torch.zeros((row,col,self.true_label.shape[0])).float()
            self.T_mat_true = torch.zeros((row,col,self.true_label.shape[0])).float()
            idx_all = np.arange(self.true_label.shape[0])
            for i in range(row):
                for j in range(col):
                    self.T_mat_true[i][j] = ((self.true_label == i) * (self.label == j)) * 1.0
                    self.T_mat[i][j] = ((self.distilled_label == i) * (self.label == j) * (self.distilled_weight > 0.0)) * 1.0


        else:
            self.num_classes = len(self.chosen_classes)
            chosen_class = torch.tensor(self.chosen_classes)
            idx = torch.nonzero( self.label.unsqueeze(-1) == chosen_class )
            self.data = self.data[ idx[:,0].view(-1) ]
            self.label = idx[:,1].view(-1).clone() #convert the chosen indices to 0,1,...
            self.true_label = idx[:,1].view(-1).clone()

        assert isinstance(self.label, torch.Tensor)

        self._load_meta()


    def load_label_multiple(self):
        '''
        I adopt .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        #NOTE presently only use for load manual training label
        noise_label = torch.load(self.label_file_path)
        assert isinstance(noise_label, dict)
        clean_label = noise_label['clean_label_train']
        noisy_label = noise_label['noise_label_train']

        if self.sample_weight_path == 'None' or self.sample_weight_path is None:
            distilled_label = clean_label.clone() # include the whole dataset, not-distilled samples have weight 0
            if self.beta_path in ['None'] or self.beta_path is None:
                sel_idx = np.arange(clean_label.shape[0])
                distilled_label = clean_label.clone()
                distilled_idx = np.random.choice(range(len(sel_idx)), size = int(len(sel_idx)//self.ratio_true), replace = False, p = None)
                distilled_weight = torch.tensor(np.zeros_like(sel_idx)) * 1.0 
                distilled_weight[distilled_idx] = 1.0
            else:
                beta_tmp = torch.load(self.beta_path)
                distilled_idx = beta_tmp['idx']
                distilled_weight[distilled_idx] = beta_tmp['beta']
        else:
            data_load = torch.load(self.sample_weight_path)
            distilled_weight = torch.tensor(data_load['distilled_weight'], dtype = torch.float)
            if not (self.beta_path in ['None'] or self.beta_path is None):
                beta_tmp = torch.load(self.beta_path)
                distilled_idx = beta_tmp['idx']
                distilled_weight[distilled_idx] = beta_tmp['beta']
            if 'self.distilled_label' in data_load.keys():
                distilled_label = torch.tensor(data_load['self.distilled_label'], dtype = torch.long)
            else:
                distilled_label = torch.tensor(data_load['distilled_label'], dtype = torch.long)


        return clean_label, noisy_label, distilled_label, distilled_weight



    def load_label_binary(self):
        '''
        I adopt .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        #NOTE presently only use for load manual training label
        noise_label = torch.load(self.label_file_path)
        assert isinstance(noise_label, dict)
        clean_label = noise_label['clean_label_train']
        noisy_label = noise_label['noise_label_train']
        if 'sel_class' in noise_label.keys():
            sel_class = noise_label['sel_class']
            sel_idx = noise_label['sel_idx']
            distilled_weight = torch.tensor(np.zeros_like(sel_idx)) * 1.0 
            if self.sample_weight_path == 'None' or self.sample_weight_path is None:
                distilled_label = clean_label.clone() # include the whole dataset, not-distilled samples have weight 0
                if self.beta_path in ['None'] or self.beta_path is None:
                    distilled_idx = np.random.choice(range(len(sel_idx)), size = int(len(sel_idx)//self.ratio_true), replace = False, p = None)
                    distilled_weight[distilled_idx] = 1.0
                else:
                    beta_tmp = torch.load(self.beta_path)
                    distilled_idx = beta_tmp['idx']
                    distilled_weight[distilled_idx] = beta_tmp['beta']
            else:
                data_load = torch.load(self.sample_weight_path)
                distilled_weight = torch.tensor(data_load['distilled_weight'], dtype = torch.float)
                if not (self.beta_path in ['None'] or self.beta_path is None):
                    beta_tmp = torch.load(self.beta_path)
                    distilled_idx = beta_tmp['idx']
                    distilled_weight[distilled_idx] = beta_tmp['beta']
                if 'self.distilled_label' in data_load.keys():
                    distilled_label = torch.tensor(data_load['self.distilled_label'], dtype = torch.long)
                else:
                    distilled_label = torch.tensor(data_load['distilled_label'], dtype = torch.long)
        else:
            sel_class = [0,1,2,3,4,5,6,7,8,9]
            sel_idx = np.arange(clean_label.shape[0])
            distilled_label = clean_label.clone()
            distilled_idx = np.random.choice(range(len(sel_idx)), size = int(len(sel_idx)//self.ratio_true), replace = False, p = None)
            distilled_weight = torch.tensor(np.zeros_like(sel_idx)) * 1.0 
            distilled_weight[distilled_idx] = 1.0

        return clean_label, noisy_label, sel_class, sel_idx, distilled_label, distilled_weight



class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    num_classes = 100


class CIFAR100CAL(CIFAR10CAL):
    """
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    num_classes = 100



if __name__ == "__main__":
    pass