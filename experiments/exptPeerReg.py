import os
import sys
import time
import numpy as np
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import options
import models
import datamanage.peerdatasets as datasets

from utils.functions import *
from experiments.baseexperiment import BaseExperiment




class ExptPeerRegC10CAL(BaseExperiment):
    def __init__(self, param_dict=None, json_path=None, outputfile=None):
        """
        Coordinate data loading, model building, training & validation
        """
        super(ExptPeerRegC10CAL, self).__init__(param_dict, json_path, outputfile)
        self._name = "ExptPeerRegC10CAL"
        self.loss_mean_all = torch.zeros((len(self.opt.chosen_classes),len(self.opt.chosen_classes))).float().to(self._device)




    def set_model(self):
        model = models.PeerRegModelCAL(self.logger.log_path, self.opt, noisy_prior = self.noisy_prior, T_mat = self.T_mat, T_mat_true = None, P_y_distill = self.P_y_distill)
        # T_mat: N*N*size, taking value 1 or 0
        if self.opt.is_load:
            last_epoch, last_val_acc = model.load(model_path = self.opt.model_path)
            self._init_epoch = last_epoch + 1
            self._last_val_acc = last_val_acc
        return model
    

    def confg_dataset(self):
        modeldataset = datasets.CIFAR10CAL
        data_root = "datasets/"
        dataset_path = os.path.join(data_root, "cifar-10-batches-py")
        datasize = 50000
        train_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4882, 0.4465], 
                                 std =[0.2023, 0.1994, 0.2010])
        ])
        test_trans = transforms.Compose([
            transforms.Normalize(mean=[0.4914, 0.4882, 0.4465], 
                                 std =[0.2023, 0.1994, 0.2010])
        ])
        return modeldataset, data_root, dataset_path, datasize, train_trans, test_trans


    def load_data(self, noise_label_fname=None):
        '''
        load datasets:
        --- training set
        --- (optional) validate set, which is part of the training set
        --- test set, which is used as clean validate set presently
        '''
        print('Preparing train, val, test dataloader ...', flush=True, file=self._outfile)

        modeldataset, data_root, dataset_path, datasize, train_trans, test_trans = self.confg_dataset()

        # noise labels setting
        if noise_label_fname is None:
            path_list = [dataset_path, "noise_label", self.opt.noise_label_fname]
        else:
            path_list = [dataset_path, "noise_label", noise_label_fname]
        label_file_path = os.path.join(*path_list) if self.opt.with_noise else None

        # validation setting; val_idx
        train_idx = None
        if self.opt.is_validate:
            idx_bound = int( (1. - self.opt.val_ratio) * datasize )
            train_idx = [i for i in range(idx_bound)]
            val_idx = [i for i in range(idx_bound, datasize)]

        # Train & peer
        train_dataset = modeldataset(root=data_root, is_train=True,
                                     transform=train_trans,
                                     label_file_path=label_file_path,
                                     selected_idx=train_idx,
                                     sample_weight_path = self.opt.sample_weight_path,
                                     beta_path = self.opt.beta_path,
                                     ratio_true = self.opt.ratio_true,
                                     chosen_classes=self.opt.chosen_classes,
                                     is_download=True)
        if '_CAL' in self.opt.lossfunc: 
            self.noisy_prior = train_dataset.noisy_prior   # cifar10 CAL
            print(f'Use the original prior: {self.noisy_prior}')
        else:
            a = train_dataset.noisy_prior
            self.noisy_prior = list(np.sqrt(a)/np.sum(np.sqrt(a))) # for cifar10 Est \hat D
            print(f'Use sqrt prior: {self.noisy_prior}')


        # load weight and distilled samples here

        self.distilled_label = train_dataset.distilled_label
        self.label = train_dataset.label
        self.true_label = train_dataset.true_label_d
        self.beta = train_dataset.distilled_weight # load from .pt files, set as default if no file provided
        self.distilled_weight = (self.beta > 0.0) * 1.0
        T_mat = train_dataset.T_mat
        # T_mat_true = train_dataset.T_mat_true
        P_y_distill = torch.tensor([torch.sum((self.distilled_label == i) * (self.distilled_weight == 1.0)) for i in range(len(self.opt.chosen_classes))]).float()
        P_y_distill /= torch.sum(P_y_distill)
        for i in range(len(self.opt.chosen_classes)):
            for j in range(len(self.opt.chosen_classes)):
                weight_sum = np.max((torch.sum(self.distilled_weight * (self.distilled_label==i)),1.0))                    
                T_mat[i,j] =  (T_mat[i,j] - torch.sum(T_mat[i,j])/weight_sum) * self.distilled_weight * (self.distilled_label==i)
                # T_mat_true[i,j] =  (T_mat_true[i,j] - torch.sum(T_mat_true[i,j])/torch.sum(self.true_label==i)) * (self.true_label==i)
        self.T_mat = T_mat.detach()
        # self.T_mat_true = T_mat_true.detach()
        self.P_y_distill = P_y_distill.detach()






        self.train_datasize = len(train_dataset)
        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.opt.batch_size,
                                       shuffle=True,
                                       num_workers=self.opt.num_workers)
        # distill
        self.distilled_weight_new = np.zeros(self.distilled_weight.shape[0]) * 1.0
        self.distilled_label_new = np.zeros(self.distilled_label.shape[0])


        # Validate
        if self.opt.is_validate:
            val_dataset   = modeldataset(root=data_root, is_train=True,
                                         transform=test_trans,
                                         label_file_path=label_file_path,
                                         selected_idx=val_idx,
                                         chosen_classes=self.opt.chosen_classes,
                                         is_download=True)
            self.val_datasize   = len(val_dataset)
            self.val_loader   = DataLoader(dataset=val_dataset,
                                           batch_size=self.opt.batch_size//4,
                                           shuffle=False,
                                           num_workers=self.opt.num_workers)

        # Test
        test_dataset  = modeldataset(root=data_root, is_train=False,
                                     transform=test_trans,
                                     chosen_classes=self.opt.chosen_classes,
                                     is_download=True)
        self.test_datasize  = len(test_dataset)
        self.test_loader  = DataLoader(dataset=test_dataset,
                                       batch_size=self.opt.batch_size//4,
                                       shuffle=False,
                                       num_workers=self.opt.num_workers)
    
    def train_epoch(self, i_epoch):
        '''
        train the model for a single epoch
        '''
        is_check_loss_dist = self.opt.is_check_loss_dist
        x_peer = None


        self.model.distilled_weight_new = self.distilled_weight_new.copy()
        self.model.distilled_label_new = self.distilled_label_new.copy()
        self.model.noise_prior_cnt = np.array(self.noisy_prior)*self.train_datasize
        self.model.noise_change_cnt = np.zeros_like(self.model.noise_prior_cnt)
        loss_rec_all = torch.zeros((len(self.opt.chosen_classes),len(self.opt.chosen_classes))).to(self._device) * 1.0
        # loss_rec_all_true = torch.zeros((len(self.opt.chosen_classes),len(self.opt.chosen_classes))).to(self._device) * 1.0

        T_mat_indicator_sum = torch.max(torch.sum(torch.sum(torch.sum(self.T_mat>0.0,2),1).view(self.T_mat.shape[0],1,-1).repeat(1,self.T_mat.shape[0],1),2) * 1.0, torch.ones((self.T_mat.shape[0],self.T_mat.shape[1]))).to(self._device)
        # T_mat_indicator_true_sum = torch.sum(torch.sum(torch.sum(self.T_mat_true>0.0,2),1).view(self.T_mat_true.shape[0],1,-1).repeat(1,self.T_mat_true.shape[0],1),2).to(self._device)
        
        for i_batch , (inputs, labels, true_labels, raw_idx) in enumerate(self.train_loader):
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            true_labels = true_labels.to(self._device)
            distilled_labels = self.distilled_label[raw_idx].to(self._device)
            distilled_weights = self.beta[raw_idx].to(self._device)
            bsize  = inputs.shape[0]


            #prepare peer terms inputs (another way)
            if self.opt.is_peerloss:
                x_peer = inputs

            # update model param for one batch

            self.model.update(inputs, labels, true_labels, x_peer, 
                              probe=self.logger.probe, is_check_loss_dist=is_check_loss_dist, raw_idx = raw_idx, loss_mean_all = self.loss_mean_all, distilled_labels = distilled_labels, distilled_weights = distilled_weights, loss_mean_all_true = None)

            if '_CAL' in self.opt.lossfunc: 
                loss_rec_all += self.model.loss_rec_all
            # display loss and corrects info within an epoch
            self.logger.display_acc_loss_batch(i_batch=i_batch, batch_size=bsize, tag_prefix="train")
            # break
        if '_CAL' in self.opt.lossfunc: 
            self.loss_mean_all = loss_rec_all / (T_mat_indicator_sum * 1.0)

            


        # distill
        if i_epoch in [65] and '_CAL' not in self.opt.lossfunc: 
            torch.save({'distilled_weight':self.model.distilled_weight_new, 
                        'distilled_label':self.model.distilled_label_new}, f'sieve_{i_epoch}_{self.opt.exp_name}.pt')
        
        # log and display info
        self.logger.log_acc_loss(tag_prefix="train")
 





class ExptPeerRegC100CAL(ExptPeerRegC10CAL):
    def __init__(self, param_dict=None, json_path=None, outputfile=None):
        """
        Coordinate data loading, model building, training & validation
        """
        super(ExptPeerRegC10CAL, self).__init__(param_dict, json_path, outputfile)
        self._name = "ExptPeerRegC100CAL"
        self.loss_mean_all = torch.zeros((len(self.opt.chosen_classes),len(self.opt.chosen_classes))).float().to(self._device)


    

    def confg_dataset(self):
        modeldataset = datasets.CIFAR100CAL
        data_root = "datasets/"
        dataset_path = os.path.join(data_root, "cifar-100-python")
        datasize = 50000
        train_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                 std =[0.2675, 0.2565, 0.2761])
        ])
        test_trans = transforms.Compose([
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                 std =[0.2675, 0.2565, 0.2761])
        ])
        return modeldataset, data_root, dataset_path, datasize, train_trans, test_trans
