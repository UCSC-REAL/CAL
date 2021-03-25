import torch
import torch.nn as nn
import torch.optim as optim

import os
import warnings
import sys


from inspect import ismodule, isclass, isfunction, getfullargspec

import networks
import models.schedulers as SKD



class BaseModel(object):
    def __init__(self, log_path, opt):
        self._name = "base model"
        self.device_num = 1
        self.is_train = opt.is_train
        self.opt = opt

        self.network = None
        self.optimizer = None
        self.lossfunc = None
        self.val_lossfunc = None

        if hasattr(opt, "gpu_idx"):
            device_list = self.opt.gpu_idx.split(',')
            self._device_num = len(device_list)

        if log_path is not None:
            self._set_model_dir(log_path)
    
    def to(self, device):
        # for network
        self.network.to(device)
        # for optimizer
        for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def parallel(self):
        '''
        handle control of parallelism, presently only for network
        '''
        if self._device_num > 1:
            self.network = nn.DataParallel(self.network)

    def predict(self, x):
        '''
        interface for getting the predictions of the network
        ------
        this is a basic implementation; extension may include calculating val loss
        '''
        self.network.eval()
        return self.network(x)

    def update(self, x, labels):
        r'''update model parameters for one step'''
        raise NotImplementedError

    def set_model(self):
        r'''define the model based on network, loss & optim'''
        raise NotImplementedError

    def set_network(self, net_arch, opt, **kwargs):
        r'''
        automatically setup a network based on net_arch and
        inputs form opt & kwargs
        '''
        assert net_arch in networks.__all__, f"Note valid ARCH {net_arch},"+\
                f"must be from list: {networks.__all__}"

        netobj = networks.__dict__[net_arch]
        # accept functions & classes
        assigned_vars = []
        if isclass(netobj):
            args, _, _, defaults, _, _, _ = getfullargspec(netobj.__init__)
            assigned_vars.append('self')
        elif isfunction(netobj):
            args, _, _, defaults, _, _, _ = getfullargspec(netobj)

        input_dict = {}
        if defaults is not None:
            for var, val in zip(reversed(args), reversed(defaults)):
                # note the order in the following branches
                # kwargs has the highest priority, then opt, then defaults
                if var in kwargs.keys():
                    input_dict[var] = kwargs[var]
                elif hasattr(opt, var):
                    input_dict[var] = eval(f"opt.{var}")
                else:
                    input_dict[var] = val
                assigned_vars.append(var)

        for var in list(set(args)-set(assigned_vars)):
            if var in kwargs.keys():
                input_dict[var] = kwargs[var]
            elif hasattr(opt, var):
                input_dict[var] = eval(f"opt.{var}")
            else:
                raise NameError

        return netobj(**input_dict)

    def set_optimizer(self, optim_name, opt, **kwargs):
        r'''
        return an optimizer
        Currently available optimizers: 
            SGD, ADAM, Adagrad
        '''
        available_optim = ["SGD", "Adam", "Adagrad"]
        assert optim_name in available_optim, f"Note valid ARCH {optim_name},"+\
                f"must be from list: {available_optim}"

        optimobj = eval(f"optim.{optim_name}")
        # accept classes
        assigned_vars = []
        if isclass(optimobj):
            args, _, _, defaults, _, _, _ = getfullargspec(optimobj.__init__)
            assigned_vars.append('self')

        input_dict = {}
        if defaults is not None:
            for var, val in zip(reversed(args), reversed(defaults)):
                # note the order in the following branches
                # kwargs has the highest priority, then opt, then defaults
                if var in kwargs.keys():
                    input_dict[var] = kwargs[var]
                elif hasattr(opt, var):
                    input_dict[var] = eval(f"opt.{var}")
                else:
                    if var == "lr":
                        # for the <required parameter> case; seems SGD only
                        assert isinstance(val, float)
                    input_dict[var] = val
                assigned_vars.append(var)

        for var in list(set(args)-set(assigned_vars)):
            if var == "params":
                if 'pretrain' in self.opt.exp_name and 'fix' in self.opt.exp_name:
                    input_dict[var] = self.network.fc.parameters()
                else:
                    input_dict[var] = self.network.parameters()
            elif var in kwargs.keys():
                input_dict[var] = kwargs[var]
            elif hasattr(opt, var):
                input_dict[var] = eval(f"opt.{var}")
            else:
                raise NameError

        return optimobj(**input_dict)

    def set_lossfunc(self, loss_name, opt, **kwargs):
        r'''return a loss function'''
        raise NotImplementedError

    def _set_model_dir(self, root_path):
        r"""
        Assuming root_path being model&date specified and created elsewhere 
        """
        self._ckpt = os.path.join(root_path, "checkpoints")
        if not os.path.exists(self._ckpt):
                os.makedirs(self._ckpt)

    def save(self, epoch=-1, val_acc=0., model_name=None):
        r'''
        I adopte .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        if model_name is None:
            model_name = f"model_present_best_val.pt"
        save_full_path = os.path.join(self._ckpt, model_name)

        torch.save({'val_acc': val_acc,
                    'epoch': epoch,
                    'network_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_full_path)

        print(f"model saved to {save_full_path}")

    def load(self, epoch=-1, model_name=None, model_path=None, is_predict=False):
        r'''
        I adopte .pt rather .pth according to this discussion:
        https://github.com/pytorch/pytorch/issues/14864
        '''
        if model_name is None:
            model_name = f"model_present_best_val.pt"

        # mpath = os.path.join(self._ckpt, model_name) if model_path is None \
        #             else os.path.join(model_path, model_name)
        mpath = model_path

        if not torch.cuda.is_available():
            ckpt = torch.load(mpath, map_location="cpu")
        else:
            ckpt = torch.load(mpath, map_location="cuda")

        # if continue training, keep last parameters
        if self.opt.is_continue_train:
            last_epoch = ckpt['epoch']
        else:
            last_epoch = -1

        last_val_acc = 0.
        if 'val_acc' in ckpt.keys():
            last_val_acc = ckpt['val_acc']
        elif 'acc' in ckpt.keys():
            last_val_acc = ckpt['acc']
        

        # load network epoch weight  #['state_dict']
        if self._device_num > 1:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            # for k, v in ckpt['network_state_dict'].items():
            for k, v in ckpt['state_dict'].items():  # for c1m only
                name = k[7:] # remove `module.` in dict keys
                new_state_dict[name] = v
            # load params
            self.network.load_state_dict(new_state_dict)
        else:
            # self.network.load_state_dict(ckpt['network_state_dict'])
            self.network.load_state_dict(ckpt['state_dict'])

        # load optimizer, if for training
        # if not is_predict and self.opt.is_continue_train:
        #     self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        #     # update learning rate if lr in new param is different
        #     for param_group in self.optimizer.param_groups:
        #         if param_group['lr'] != self.opt.lr:
        #             param_group['lr'] = self.opt.lr
        #             param_group['initial_lr'] = self.opt.lr

        mfolder = self._ckpt if model_path is None else model_path
        print(f"model loaded from {mfolder}")

        return last_epoch, last_val_acc
    
    def load_dmi(self, epoch=-1, model_name=None, model_path=None):

        if model_name is None:
            model_name = f"model_present_best_val.pt"

        mpath = os.path.join(self._ckpt, model_name) if model_path is None \
                    else os.path.join(model_path, model_name)

        if not torch.cuda.is_available():
            self.network = torch.load(mpath, map_location="cpu")
        else:
            self.network = torch.load(mpath)

        last_epoch = -1
        last_val_acc = 0.

        mfolder = self._ckpt if model_path is None else model_path
        print(f"model loaded from {mfolder}")

        return last_epoch, last_val_acc

    def set_lr_scheduler(self, last_epoch):
        r'''
        set up learning rate schedulers
        '''
        # 2020-03-17 delete old step lr scheduler
        # if self.opt.lr_scheduler == "step":
        #     self._lr_scheduler = SKD.torchLRSKD.StepLR(optimizer=self.optimizer, 
        #                                                step_size=self.opt.lr_decay_step_size,
        #                                                gamma = self.opt.lr_decay_rate,
        #                                                last_epoch=last_epoch)
        if self.opt.lr_scheduler == "step":
            self._lr_scheduler = SKD.StepLR(optimizer=self.optimizer, 
                                            step_size=self.opt.lr_decay_step_size,
                                            gamma = self.opt.lr_decay_rate,
                                            last_epoch=last_epoch)
        elif self.opt.lr_scheduler == "multistep":
            self._lr_scheduler = SKD.MultiStepLR(optimizer=self.optimizer,
                                                 milestones=self.opt.lr_milestones,
                                                 gamma=self.opt.lr_gamma,
                                                 last_epoch=last_epoch)
        elif self.opt.lr_scheduler == "SGDR":
            T_max = self.opt.max_epoch if self.opt.lr_T_max == 0 else self.opt.lr_T_max
            self._lr_scheduler = SKD.CosineAnnealingLR(optimizer=self.optimizer,
                                                       T_max=T_max,
                                                       eta_min=self.opt.lr_eta_min,
                                                       last_epoch=last_epoch)
        elif self.opt.lr_scheduler == "none":
            self._lr_scheduler = None
        
        else:
            warnings.warn("Not valid option for lr scheduler type, using `none` instead", UserWarning)
            self._lr_scheduler = None

    def lr_scheduler_step(self):
        if self._lr_scheduler is not None:
            if self.opt.is_peerloss:
                tmp_alpha = self.get_alpha()
                self._lr_scheduler.alpha = tmp_alpha
            self._lr_scheduler.step()
    
    def lr_scheduler_restart(self):
        if self.opt.lr_scheduler == "SGDR":
            self._lr_scheduler.restart()

    def display_lr(self):
        for param_group in self.optimizer.param_groups:
            print(f"the present lr is :{param_group['lr']:.7f}")

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]



if __name__ == "__main__":
    C = type("options", (object,), {})
    opt = C()
    setattr(opt, "netARCH", "resnet_cifar18")
    setattr(opt, "num_classes", 10)
    setattr(opt, "is_train", True)

    testrun = BaseModel(None, opt)