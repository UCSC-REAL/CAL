import os
import sys
import json
import h5py
import pandas as pd
import pickle
import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt


class Probe(object):
    r"""
    Class for recording data by specified tag
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.datafile_path = os.path.join(self.data_path, "data.pkl")

        # initialize dictionary for holding data temparaily 
        self.data_pool = {}
        self.step = -1

    def update_step(self):
        self.step += 1

    def add_data(self, tag, data, global_step=None, walltime=None):

        step = self.step if global_step is None else global_step
        if step not in self.data_pool.keys():
            self.data_pool[step] = {}

        if tag not in self.data_pool[step].keys():
            self.data_pool[step][tag] = []

        self.data_pool[step][tag] += data if isinstance(data, list) else [data]

    def cleanup_data_pool(self, step=None, key=None):
        if step is None:
            if key is None:
                del self.data_pool
                self.data_pool = {}
            else:
                del self.data_pool[self.step][key]
        else:
            if key is None:
                del self.data_pool[step]
            else:
                del self.data_pool[step][key]

    def flush(self):
        r'''flush to file & cleanup'''
        pass

class Logger(object):
    r"""
    record & process data during training;
    ---------
    It is designed that the raw data recording is completely handled by self.probe, 
    and all relevant computation methods are nested in this Logger.
    This is for the purpose of separating the data processing from the training process.
    """
    def __init__(self, opt, json_path, outputfile=None):
        self._name = "logger"
        self.log_root = opt.log_root
        self._plot_dir = "plots"
        self._data_dir = "data"
        self.opt = opt

        if outputfile is None:
            outputfile = sys.stdout
        self._outfile = outputfile

        self._dump_info()
        self._set_directories(json_path)

        # initialize data probe for recording data
        self.probe = Probe(self._data_path)
        self.flush_step = 10
        self.param = {}
        self.record = {}
        self.plot_step = [i for i in range(50)] +\
                         [i for i in range(50, 100, 5)] +\
                         [i for i in range(100, 200, 10)] +\
                         [i for i in range(200, 300, 20)] +\
                         [i for i in range(300, 400, 20)]

    @property
    def logdir(self):
        return self.log_path
    
    def _dump_info(self):
        r"""
        print experiment info
        """
        print(f"\n{'Dataset:':>10} {self.opt.dataset}"\
              f"\n{'NetATCH:':>10} {self.opt.netARCH}"\
              f"\n{'Optimizer:':>10} {self.opt.optimizer}"\
              f"\n{'Lossfunc:':>10} {self.opt.lossfunc}",\
              flush=True, file=self._outfile)

    def _set_directories(self, json_path):
        r"""
        * if given json file, means either continue training or 
            load trained model for validation. Use it as log_path
        * if json_path == None, create new log_path

        log_path will be the directory for storing:
            * experiment options
            * trained model (checkpoint)
            * results: figures & data
        """
        # set log root dir
        if json_path is None:
            # for new experiment, create relevant directories
            dir_name_list = [self.log_root,
                             self.opt.dataset,
                             self.opt.netARCH,
                             self.opt.exp_name]
            temp_str = os.path.join(*dir_name_list)
            self.log_path = f"{temp_str}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"

            #resolve possible conflict
            if os.path.exists(self.log_path):
                old_path = self.log_path
                rand = np.random.RandomState()
                self.log_path = f"{old_path}_{rand.randint(99999):0>5}"

            os.makedirs(self.log_path)

        else:
            # if valid json_path, using existed directories
            self.log_path = json_path
            if not os.path.exists(self.log_path):
                raise ValueError(f"required path,{json_path}, not exist")

        # set plot dir & data dir
        self._plot_path = self.built_path(self.log_path, self._plot_dir)
        self._data_path = self.built_path(self.log_path, self._data_dir)

        print("\n{}".format(12*"---"), flush=True, file=self._outfile)
        print(f"Working at log_path: {self.log_path}", flush=True, file=self._outfile)
    
    def built_path(self, *args):
        r"""
        build path based on the args input
        """
        path = os.path.join(*args)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def add_param(self, tag, val):
        r"""
        interface for update/record parameters; used for later computation
        """
        self.param[tag] = val
    
    def add_record(self, tag, val):
        r"""
        interface for updating recorded data
        """
        if tag not in self.record.keys():
            self.record[tag] = []
        self.record[tag] += val if isinstance(val, list) else [val]

    def add_data(self, tag, data, global_step=None, walltime=None):
        r"""
        interface for recording raw data
        """
        self.probe.add_data(tag, data, global_step, walltime)

    def display_acc_loss_batch(self, i_batch, batch_size, tag_prefix="train"):
        r"""
        display batch corrects & loss during training
        ---------
        relevant data:
            {tag_prefix}/loss
            {tag_prefix}/corrects
            {tag_prefix}/batch_num
        """
        if i_batch == 0:
            self.p = 0.1

        total_num_batch = self.param[f"{tag_prefix}/batch_num"]
        if i_batch >= int(self.p * total_num_batch):

            # total_loss += loss * float(bsize) / float(self.opt.batch_size)
            loss = self.probe.data_pool[self.probe.step][f"{tag_prefix}/loss"][-1]
            corrects = self.probe.data_pool[self.probe.step][f"{tag_prefix}/corrects"][-1]

            print(f"[{i_batch+1:3}/{total_num_batch:3}], current batch loss: {loss:.5f}",\
                f"with corrects {corrects:3.0f}/{batch_size} ({corrects/float(batch_size):.5f})",\
                flush=True, file=self._outfile)
            
            self.p += 0.1

    def log_acc_loss(self, tag_prefix="train"):
        r"""
        compute & record average accuracy & loss for each epoch
        ---------
        relevant data:
            {tag_prefix}/corrects
            {tag_prefix}/loss
            {tag_prefix}/datasize
            {tag_prefix}/batch_num
        
        recorded:
            {tag_prefix}/avg_acc
            {tag_prefix}/avg_loss
        """
        corrects = self.probe.data_pool[self.probe.step][f"{tag_prefix}/corrects"]
        loss = self.probe.data_pool[self.probe.step][f"{tag_prefix}/loss"]

        avg_acc = np.sum(corrects) / float(self.param[f"{tag_prefix}/datasize"])
        avg_loss = np.sum(loss) / float(self.param[f"{tag_prefix}/batch_num"])

        print(f"\nFor epoch {self.probe.step}, average {tag_prefix} "\
              f"loss = {avg_loss:.5f}; accuracy = {avg_acc:.5f};",\
              flush=True, file=self._outfile)

        if  f"{tag_prefix}/avg_acc" not in self.record.keys():
            self.record[f"{tag_prefix}/avg_acc"] = []
        self.record[f"{tag_prefix}/avg_acc"] += [avg_acc]

        if  f"{tag_prefix}/avg_loss" not in self.record.keys():
            self.record[f"{tag_prefix}/avg_loss"] = []
        self.record[f"{tag_prefix}/avg_loss"] += [avg_loss]

   

    def log_base_and_peer(self):
        r"""
        compute & record average base loss & peer term for each epoch
        ---------
        relevant data:
            train/base
            train/peer
            train/batch_num

        recorded:
            train/avg_base
            train/avg_peer
        """
        base = self.probe.data_pool[self.probe.step]["train/base"]
        peer = self.probe.data_pool[self.probe.step]["train/peer"]

        avg_base = np.sum(base) / float(self.param["train/batch_num"])
        avg_peer = np.sum(peer) / float(self.param["train/batch_num"])

        if "train/avg_base" not in self.record.keys():
            self.record["train/avg_base"] = []
        self.record["train/avg_base"] += [avg_base]

        if "train/avg_peer" not in self.record.keys():
            self.record["train/avg_peer"] = []
        self.record["train/avg_peer"] += [avg_peer]


    def cleanup(self, *tags):
        save_step=20
        if tags == []:
            return
        if (self.probe.step + 1) % save_step != 0:
            return

        for s in range(self.probe.step - save_step + 1, self.probe.step + 1):
            for tag in tags:
                self.probe.cleanup_data_pool(step=s, key=tag)

    def dump_logged_record(self, fname="logged_record.pkl"):
        self._save_pickle(self.record ,fname=fname)

    def dump_lr_alpha_acc_loss(self, save_step=None):
        r"""
        interface of dumping info to csv file in _data_path
        column-order (if no data, skip):
            Epoch, Learning Rate, Alpha, Train Accuracy, Validate Accuracy, Test Accuracy,
            Base Loss, Peer Term, Train Loss, Validate Loss, Test Loss
        """
        steplen = self.flush_step if save_step is None else save_step
        if (self.probe.step + 1) % steplen != 0:
            return

        keymap={
            "step": "Epoch", "lr": "Learning Rate", "alpha": "Alpha",
            "train/avg_acc": "Trian Accuracy", "train/avg_loss": "Trian Loss",
            "train/avg_base": "Base Loss", "train/avg_peer": "Peer Term",
            "train/avg_R1": "R1", "train/avg_R2": "R2",
            # "train/avg_biln_est": "biln_est", "train/avg_biln_true": "biln_true",
            "validate/avg_acc": "Validate Accuracy", "validate/avg_loss": "Validate Loss",
            "test/avg_acc": "Test Accuracy", "test/avg_loss": "Test Loss",
        }

        highlim = self.probe.step + 1
        lowlim = highlim - steplen

        cfg={}
        cfg["step"] = [e+1 for e in range(lowlim, highlim)]
        cfg["lr"] = self.record["train/lr"][lowlim:highlim]
        cfg["alpha"] = self.record["train/alpha"][lowlim:highlim]

        for surfix in ["avg_acc", "avg_base", "avg_peer", "avg_loss", "avg_R1", "avg_R2"]:
            for prefix in ["train", "validate", "test"]:
                k = f"{prefix}/{surfix}"
                if k in self.record.keys():
                    cfg[k] = self.record[k][lowlim:highlim]
        # cfg['train/avg_biln_est'] = self.record["train/avg_biln_est"][lowlim:highlim]
        # cfg['train/avg_biln_true'] = self.record["train/avg_biln_true"][lowlim:highlim]

        self._save_csv(fname="acc_loss_log.csv", keymap=keymap, **cfg)

    def _save_csv(self, fname, keymap, **kwargs):
        r"""
        method for saving to csv file in _data_path;
        recommended for relatively small data; easy to visualize & manipulate,
        e.g. train_acc, test_acc, ...
        """
        fpath = os.path.join(self._data_path, fname)

        Dict ={}
        for k in kwargs.keys():
            Dict[keymap[k]] = kwargs[k] if isinstance(kwargs[k], list) else [kwargs[k]]
        
        df = pd.DataFrame.from_dict(Dict)

        if os.path.isfile(fpath):
            df.to_csv(fpath, index=False, mode='a', header=False)
        else:
            df.to_csv(fpath, index=False, mode='w')

    def _save_pickle(self, data, fname):
        fpath = os.path.join(self._data_path, fname)
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
        print(f"data saved to {fpath}")

    def _save_h5py(self, key, val, fname):
        fpath = os.path.join(self._data_path, fname)
        with h5py.File(fpath, 'a') as hf:
            hf.create_dataset(key, data=val)
