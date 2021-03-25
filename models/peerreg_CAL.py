import os
import pandas as pd

import torch
import torch.nn as nn
import numpy as np

from .basepeer import BasePeerModel
import lossfuncs as loss
import models.schedulers as SKD

class PeerRegModelCAL(BasePeerModel):
    def __init__(self, log_path, opt, noisy_prior=None, T_mat = None, T_mat_true = None, P_y_distill = None):
        super(PeerRegModelCAL, self).__init__(log_path, opt, is_inherit=True)

        # save normal loss term & peer term
        fpath_list = [log_path, "data", "batch_base_peer_records.csv"]
        self._data_save_path = os.path.join( *fpath_list )
        self.noisy_prior = [1./opt.num_classes for i in range(opt.num_classes)] \
                            if noisy_prior is None else \
                                noisy_prior
        self.softmax = nn.Softmax(dim=-1)
        self.num_classes = opt.num_classes
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self.T_mat = T_mat.to(self._device)

        self.P_y_distill = P_y_distill.to(self._device)
        # setup model
        self.set_model()
    
    def set_lossfunc(self, loss_name, opt):
        '''
        setup the loss function for the model
        ------
        available loss functions (w or w/o peer):
            CE, BCE, Bi-Tempered, Bi-Tempered Binary
        '''
        if opt.is_peerloss:
            assert opt.peer_size == 1
        if loss_name in ['crossentropy_CAL', 'crossentropy']:
            lossfunc = nn.CrossEntropyLoss()
            self.val_lossfunc = nn.CrossEntropyLoss()
        else:
            raise ValueError('presently only support CrossEntropy')

        # presently (2020-01-15), We found CE + CEeps performs better than CE + CE
        if opt.is_peerloss:
            if not isinstance(self.noisy_prior, torch.Tensor):
                self.noisy_prior = torch.tensor(self.noisy_prior, device=self._device).unsqueeze(0)
            if loss_name in ['crossentropy_CAL']:
                lossfunc = loss.PeerLossRegCE( opt.alpha, self.noisy_prior, loss_name, T_mat = self.T_mat, T_mat_true = None, P_y_distill = self.P_y_distill)
            else:
                lossfunc = loss.PeerLossRegCE( opt.alpha, self.noisy_prior, loss_name)
            print("\nPeer Loss Function: ", lossfunc)

        return lossfunc

    def mixup(self, x, label):
        batch_size = int(x.size(0))
        idx = torch.randperm(batch_size)
        l = np.random.beta(0.5, 0.5) # C1M alpha = 0.5  
        # l = np.random.beta(4, 4) # CIFAR alpha = 4        
        l = max(l, 1-l)   
        label_x = (torch.zeros(batch_size, self.num_classes).to(self._device)).scatter_(1, label.view(-1,1), 1)
        x_a, x_b = x, x[idx]
        label_a, label_b = label_x, label_x[idx]
        mixed_x = l * x_a + (1 - l) * x_b
        mixed_label = l * label_a + (1 - l) * label_b
        return mixed_x, mixed_label, l
        
    def update(self, x, labels, true_labels=None, x_peer=None, probe=None, is_check_loss_dist=False, raw_idx = None, loss_mean_y = None, loss_mean_n = None, loss_mean_all = None, distilled_labels = None, distilled_weights = None, loss_mean_y_true = None, loss_mean_n_true = None, loss_mean_all_true = None):
        '''
        update/train the model parameters for one iteration
        '''
        # set to learnable
        self.network.train()
        with torch.set_grad_enabled(True):
            # forward
            self.optimizer.zero_grad()
            if '_mix' in self.opt.lossfunc: # not used in CAL. For future extension
                x_mix, label_input, lmd = self.mixup(x,labels)
                outputs_m = self.network(x_mix)
                outputs = self.network(x)
                outputs_mix = torch.cat((outputs_m,outputs),0)
            else:
                label_input = labels
                outputs = self.network(x)
            if self.opt.is_peerloss:
                # output_peer = self.network(x_peer)  # equivalent form
                output_peer = outputs
                if '_mix' in self.opt.lossfunc:
                    loss, base, peer = self.lossfunc(outputs_mix, label_input, outputs, labels, lmd, noisy_prior_new = torch.tensor(self.noisy_prior_new, device=self._device).unsqueeze(0), weight = self.weight[raw_idx])
                elif self.opt.lossfunc == 'crossentropy_CAL':
                    loss, base, peer, CAL_est, self.loss_rec_all = self.lossfunc(outputs, label_input, output_peer, true_y = true_labels, distill_y = distilled_labels, raw_idx = raw_idx, loss_mean_all = loss_mean_all, loss_mean_all_true = loss_mean_all_true,  distilled_weights = distilled_weights)
                else:
                    loss, base, peer = self.lossfunc(outputs, label_input, output_peer)
            else:
                loss = self.lossfunc(outputs, label_input)
                
            # record performance; taking into accout BCE case
            preds = torch.max(outputs, 1)[1] 
            labels = labels % self.num_classes
            corrects = torch.sum(preds == labels).double()

            # compute and prepare for log loss dist
            if is_check_loss_dist:
                eps = 1e-8

                out_prob = self.softmax(outputs)
                out_prob_p = out_prob[:,1]
                probs_label = out_prob.gather(1, (labels).view(-1,1)).view(-1)
                log_out_prob = -torch.log(out_prob + eps)
                ce_term = -torch.log(probs_label + eps)
                norm_term = torch.mean(log_out_prob, dim=1) 



                # distill (sieve) samples (general case)
                idx_true = (ce_term - norm_term < -8.0).detach()
                idx_false = (ce_term - norm_term > -8.0).detach()
                self.distilled_weight_new[raw_idx[idx_true]] = 1.0
                self.distilled_weight_new[raw_idx[idx_false]] = 1.0
                self.distilled_label_new[raw_idx[idx_true]] = labels[idx_true].cpu().numpy().astype(int)
                self.distilled_label_new[raw_idx[idx_false]] = preds[idx_false].cpu().numpy().astype(int)



                
            # collect data; remove detach() as we only record item()
            if probe is not None:
                probe.add_data("train/corrects", corrects.item())
                probe.add_data("train/loss", loss.item() * x.shape[0] / self.opt.batch_size )

                        


            # backprop
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    pass