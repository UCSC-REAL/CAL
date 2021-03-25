import os
import warnings
import pandas as pd

import torch
import torch.nn as nn
import numpy as np

from .basemodel import BaseModel
import lossfuncs as loss
import models.schedulers as SKD

class BasePeerModel(BaseModel):
    def __init__(self, log_path, opt, is_inherit=False):
        super(BasePeerModel, self).__init__(log_path, opt)

        # save normal loss term & peer term
        fpath_list = [log_path, "data", "batch_base_peer_records.csv"]
        self._data_save_path = os.path.join( *fpath_list )

        # setup model
        if not is_inherit:
            self.set_model()

    def set_model(self):
        r'''
        this is a basic way for setting model, extension is possible
        e.g. with multiple networks, optimizers or loss functions,
        modify this method accordingly
        ------
        method set_network & set_optimizer is defined in basemodel.py
        see the implementations for detail
        '''
        if self.opt.dataset == "Clothing1M":
            assert self.opt.netARCH == "resnet50"
            self.network   = self.set_network(self.opt.netARCH, self.opt, pretrained=True)
        elif 'pretrain' in self.opt.exp_name:
            assert self.opt.netARCH == "resnet18"  # only test this case
            self.network   = self.set_network(self.opt.netARCH, self.opt, pretrained=True)
            if 'fix' in self.opt.exp_name:
                for param in self.network.parameters():
                    param.requires_grad = False
            else:
                for param in self.network.parameters():
                    param.requires_grad = True
            num_ftrs = self.network.fc.in_features
            self.network.fc = nn.Linear(num_ftrs, self.opt.num_classes)
        else:
            self.network   = self.set_network(self.opt.netARCH, self.opt)
        # if 'pretrain' in self.opt.exp_name and 'fix' in self.opt.exp_name:
        #     self.optimizer = self.set_optimizer(self.opt.optimizer, self.opt)
        self.optimizer = self.set_optimizer(self.opt.optimizer, self.opt)
        self.lossfunc  = self.set_lossfunc(self.opt.lossfunc, self.opt)
    
    def set_lossfunc(self, loss_name, opt):
        '''
        setup the loss function for the model
        ------
        available loss functions (w or w/o peer):
            CE, BCE, Bi-Tempered, Bi-Tempered Binary
        '''
        if loss_name == 'crossentropy':
            if opt.is_peerloss and opt.peer_size == 1:
                lossfunc = loss.CrossEntropyLossStable()
            else:
                lossfunc = nn.CrossEntropyLoss()
            self.val_lossfunc = nn.CrossEntropyLoss()
        
        elif loss_name == 'bcewithlogits':
            if opt.is_peerloss and opt.peer_size == 1:
                lossfunc = loss.BCEwithLogitsLossStable()
                # lossfunc = nn.BCEWithLogitsLoss()
            else:
                lossfunc = nn.BCEWithLogitsLoss()
            self.val_lossfunc = nn.BCEWithLogitsLoss()

        elif loss_name == 'bitempered':
            lossfunc = loss.SparseBiTemperedLogisticLoss(opt.T1, opt.T2)
            self.val_lossfunc = loss.SparseBiTemperedLogisticLoss(opt.T1, opt.T2)

        elif loss_name == 'bitempered_binary':
            lossfunc = loss.BiTemperedBinaryLogisticLoss(opt.T1, opt.T2)
            self.val_lossfunc = loss.BiTemperedBinaryLogisticLoss(opt.T1, opt.T2)

        else:
            raise ValueError('presently only support CrossEntropy')

        # presently (2020-01-15), We found CE + CEeps performs better than CE + CE
        if opt.is_peerloss:
            if opt.peer_size == 1:
                if opt.lossfunc == 'crossentropy':
                    baseloss = lossfunc
                    lossfunc = loss.PeerLossOneCE( baseloss, opt.alpha )
                elif opt.lossfunc == 'bcewithlogits':
                    baseloss = lossfunc
                    lossfunc = loss.PeerLossOneBCE( baseloss, opt.alpha )
                else:
                    baseloss = lossfunc
                    lossfunc = loss.PeerLossOne( baseloss, opt.alpha )
            else:
                assert opt.lossfunc == 'crossentropy', "peersize > 1 only for CE"
                lossfunc = loss.PeerLoss( opt )
            print("\nPeer Loss Function: ", lossfunc)

        return lossfunc
    
    def predict(self, x, labels=None):
        '''
        interface for getting the predictions;
        For monitor the evolution of our model, we return the loss as well
        '''
        self.network.eval()
        outputs = self.network(x)
        loss = None
        if (labels is not None) and (self.val_lossfunc is not None):
            loss = self.val_lossfunc(outputs, labels.float()) if len(outputs.shape)==1 \
                        else self.val_lossfunc(outputs, labels)
        return outputs, loss

    def update(self, x, labels, true_labels=None, x_peer=None, label_peer=None, probe=None):
        '''
        update/train the model parameters for one iteration
        '''
        # set to learnable
        self.network.train()
        with torch.set_grad_enabled(True):
            # forward
            self.optimizer.zero_grad()
            outputs = self.network(x)
            if self.opt.is_peerloss:
                output_peer = self.network(x_peer)
                loss, base, peer = self.lossfunc(outputs, labels.float(), output_peer, label_peer.float()) \
                        if len(outputs.shape)==1 \
                        else self.lossfunc(outputs, labels, output_peer, label_peer)
            else:
                loss = self.lossfunc(outputs, labels.float()) if len(outputs.shape)==1 \
                        else self.lossfunc(outputs, labels)

            # record performance; taking into accout BCE case
            preds = (outputs.detach() > 0.).long() if len(outputs.shape)==1 \
                    else torch.max(outputs.detach(), 1)[1] 
            corrects = torch.sum(preds == labels).double()

            if probe is not None:
                probe.add_data("train/corrects", corrects.detach().item())
                probe.add_data("train/loss", loss.detach().item() * x.shape[0] / self.opt.batch_size )
                probe.add_data("train/conf_mat", [(l.detach().item(),p.item()) for l, p in zip(labels, preds)])
                if self.opt.is_peerloss:
                    probe.add_data("train/base", base.detach().item() * x.shape[0] / self.opt.batch_size)
                    probe.add_data("train/peer", peer.detach().item() * x.shape[0] / self.opt.batch_size)

            # backprop
            loss.backward()
            self.optimizer.step()

    def set_alpha_scheduler(self, last_epoch):
        '''
        set up alpha scheduler
        '''
        if self.opt.alpha_scheduler == "step":
            self._alpha_scheduler = \
                SKD.StepAlpha(self.lossfunc, self.opt.alpha_step_size, self.opt.gamma)

        elif self.opt.alpha_scheduler == "multistep":
            self._alpha_scheduler = \
                SKD.MultiStepAlpha(self.lossfunc, self.opt.milestones, self.opt.gamma)

        elif self.opt.alpha_scheduler == "cosanneal":
            self._alpha_scheduler = \
                SKD.CosAnnealingAlpha(self.lossfunc, self.opt.T_max, self.opt.eta_min)

        elif self.opt.alpha_scheduler == "seg":
            # assert all(ele <= 1.0 for ele in self._opt.alpha_list)
            self._alpha_scheduler = \
                SKD.SegAlpha(self.lossfunc, self.opt.alpha_list, self.opt.milestones, last_epoch)

        elif self.opt.alpha_scheduler == "none":
            self._alpha_scheduler = None
        
        else:
            warnings.warn("Not valid option for alpha scheduler type, using `none` instead", UserWarning)
            self._alpha_scheduler = None

    def alpha_scheduler_step(self):
        if self._alpha_scheduler is not None:
            # Raise a warning if has alpha scheduler and it proceed after learning rate scheduler
            if self._alpha_scheduler._step_count == 0:
                if self._lr_scheduler is not None and self._lr_scheduler._step_count > 0:
                    warnings.warn("`_alpha_scheduler.step()` should proceed before `_lr_scheduler.step()`, "
                                "so that lr can be properly rescaled by the present alpha", UserWarning)
            self._alpha_scheduler.step()

    def display_alpha(self):
        print(f"the present alpha for Peer Loss is {self.lossfunc._alpha:.7f}")

    def get_alpha(self):
        if self.opt.is_peerloss:
            alpha = self.lossfunc._alpha
        else:
            alpha = 0.
        return alpha


if __name__ == "__main__":
    pass