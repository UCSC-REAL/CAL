import os
import sys
import time
import numpy as np
import random

import torch
import torchvision

import options
import models
import datamanage.peerdatasets as datasets
from utils.newlogger import Logger
from utils.functions import *


class BaseExperiment(object):
    def __init__(self, param_dict=None, json_path=None, outputfile=None):
        """
        Coordinate data loading, model building, training & validation
        """
        self._name = "BaseExperiment"
        self._save_step = 30
        self._init_epoch = 0
        self._outfile = sys.stdout if outputfile is None else open(outputfile, "w")

        # get input parameters & check integrity
        # func 'get_parameter'; from utils
        self.opt = self.get_option(param_dict, json_path)

        # setup device for running
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_idx
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        print(f"{'device:':>10} {self._device}", flush=True, file=self._outfile)

        # create logger & record the parameters
        # --- setting a unique name for present expt: exp_name + data_time
        # --- create corresponding directories
        # --- save the model options if not loaded from json file
        self.logger = Logger(self.opt, json_path, self._outfile)
        if json_path is None:
            save_opt_to_json(self.opt, self.logger.log_path)
        
        # prepare dataloaders; set before building model
        self.load_data()

        # record dataloader related params; after load_data()
        self.register_param()

        # prepare the model: initialize a new model for training; or
        # load the trained model for validation
        self.model = self.set_model()
        if torch.cuda.is_available():
            print(f"Using GPU: {self.opt.gpu_idx}", flush=True, file=self._outfile)
            self.model.parallel()
        self.model.to(self._device)
    
    def get_option(self, param_dict, json_path):
        r"""options.PeerOptions() subjetcs to change, depending on used model"""
        return get_parameters(options.PeerOptions(), param_dict, json_path)
    
    def set_model(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def register_param(self):
        attr_list = ["train_loader", "train_datasize", "test_loader", "test_datasize"]
        attr_list += ["val_loader", "val_datasize"] if self.opt.is_validate else []
        for attr in attr_list:
            assert hasattr(self, attr)

        # initialize logger attributes
        self.logger.add_param("train/datasize", self.train_datasize)
        self.logger.add_param("train/batch_num", len(self.train_loader))
        self.logger.add_param("test/datasize", self.test_datasize)
        self.logger.add_param("test/batch_num", len(self.test_loader))
        if self.opt.is_validate:
            self.logger.add_param("validate/datasize", self.val_datasize)
            self.logger.add_param("validate/batch_num", len(self.val_loader))

    def train(self):
        if not self.opt.is_train:
            raise ValueError('train_model only applie for _is_train=True')

        print(f"\nBegin training at {time.asctime()}", flush=True, file=self._outfile)
        t_begin = time.time()

        # set up lr schedualer for update learning rate during training.
        # see https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR
        self.model.set_lr_scheduler( self._init_epoch - 1 )

        # set up alpha scheduler for Peer Loss
        if self.opt.is_peerloss:
            self.model.set_alpha_scheduler( self._init_epoch - 1 )

        # update maximum clean validate accuracy
        val_acc_max = 0.
        if self.opt.is_load:
            val_acc_max = self._last_val_acc
        self.weight = np.ones(self.train_datasize)
        self.model.noisy_prior_new = self.noisy_prior.copy()
        for i_epoch in range(self._init_epoch, self.opt.max_epoch):

            print('\n{}'.format(12*'------'), flush=True, file=self._outfile)

            # update logger & record alpha and lr (NOTE need to record before train_epoch) 
            self.logger.probe.update_step()
            self.logger.add_record("train/alpha", self.model.get_alpha())
            self.logger.add_record("train/lr", self.model.get_lr())

            # train one epoch
            self.train_epoch(i_epoch)

            # monitor clean/noise validate performance
            nv_acc  = None
            if self.opt.is_validate:
                nv_acc = self.validate(i_epoch, self.val_loader, self.val_datasize, tag_prefix="validate")
            v_acc = self.validate(i_epoch, self.test_loader, self.test_datasize, tag_prefix="test")

            # update alpha in Peer Loss; used before updating lr
            if self.opt.is_peerloss:
                self.model.display_alpha()
                self.model.alpha_scheduler_step()

            # update lr using schduler; scaled by alpha if with peerloss
            # see https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR
            self.model.display_lr()
            if self.opt.lr_scheduler == "SGDR":
                if (i_epoch+1) % (self.opt.lr_T+1) == 0:
                    self.model.lr_scheduler_restart()
            self.model.lr_scheduler_step()

            # saving model for selected epochs
            acc_record = nv_acc if self.opt.is_validate else v_acc
            if acc_record > val_acc_max:
                val_acc_max = acc_record
                self.model.save(i_epoch, acc_record)

            # dump lr, alpha, accuracy and loss (train/val/test) info at certain stepsize
            self.logger.dump_lr_alpha_acc_loss()

            # output training information
            t_end = time.time() # end t for this epoch
            print("{0}\n".format(12*'------')+\
                  f"time cost for this output period: {t_end - t_begin:.3f}(s)",\
                  flush=True, file=self._outfile)
            t_begin = time.time() # starting t for next epoch

        # dump logged record
        self.logger.dump_logged_record()

        print('{0}training end{0}'.format(8*"---"), flush=True, file=self._outfile)
        if self._outfile is not sys.stdout:
            self._outfile.close()

    def train_epoch(self, i_epoch):
        raise NotImplementedError

    def validate(self, i_epoch, val_loader, val_datasize, tag_prefix):
        '''
        validate the trained model's performance;
        --------------------------------------------------------------------------------
        Presently validate all along the training, checking the evolution of the model.
        So that we load validate data together with the training data in self.train method.
        '''
        for i_batch , (inputs, labels, true_labels, noise_mark) in enumerate(val_loader):
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            bsize  = inputs.shape[0]

            outputs, loss = self.model.predict(inputs, labels)


            preds = (outputs > 0.0).long() if len(outputs.shape)==1 \
                    else torch.max(outputs, 1)[1]

            corrects = torch.sum(preds == labels).double()

            # register data
            self.logger.add_data(f"{tag_prefix}/corrects", corrects.item())
            self.logger.add_data(f"{tag_prefix}/loss", loss.item())


        # log and display info
        self.logger.log_acc_loss(tag_prefix=tag_prefix)
        return self.logger.record[f"{tag_prefix}/avg_acc"][-1]



if __name__ == "__main__":
    pass