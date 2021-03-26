import torch
import numpy as np
import random
from experiments import ExptPeerRegC10CAL

seed = 10086
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

LossAbbr = {
    "crossentropy": "CE",
    "crossentropy_CAL": "CE_CAL",
}


#for the complete list of options, see utils/options.py

dataset = "CIFAR10"
netARCH = "resnet_cifar34"




#-------------- customized parameters --------------#
noise_rate = 0.2

# lossfunc = "crossentropy"
lossfunc = "crossentropy_CAL"

gpu_idx = "0"
#---------------------------------------------------#






max_epoch = 100

outfile = None
json_path = None
is_peer = True # by default peersize=1
with_noise = True
noise_file = f"IDN_{noise_rate}_C10.pt"
chosen_classes = [0,1,2,3,4,5,6,7,8,9]




# tune alpha_list if necessary
if lossfunc == "crossentropy":
    alpha_list = [0.0, 2.0, 2.0] #  for sample selection
    milestones = [10, 40, 80]
    sample_weight_path = None
elif lossfunc == "crossentropy_CAL":
    alpha_list = [0.0, 1.0, 1.0] # for CAL
    milestones = [10, 40, 80]
    sample_weight_path = f'sieve_65_CE_{dataset}_{noise_rate}.pt'
else:
    ValueError('Undefined loss functions')

# Generate expt name
exp_mark = f'{dataset}_{noise_rate}'
exp_name = f"{LossAbbr[lossfunc]}_{exp_mark}"

if __name__ == "__main__":
    # Main
    Run = ExptPeerRegC10CAL(
        {"--is_train": True,
        "--is_plot_results": False,
        "--is_class_resolved": False,
        "--is_load": False,
        "--exp_name": exp_name,
        "--dataset": dataset,
        "--netARCH": netARCH,
        "--num_classes": 10,
        "--lossfunc": lossfunc,
        "--optimizer": "SGD",
        "--lr": 0.1, # 0.1
        "--lr_scheduler": "step",
        "--weight_decay": 0.0005,
        "--lr_decay_step_size": 60, # 60
        "--lr_decay_rate": 0.1, # 0.1
        "--batch_size": 128,
        "--max_epoch": max_epoch,
        "--is_validate": False,
        "--val_ratio": 0.0,
        "--with_noise": with_noise,
        "--noise_label_fname": noise_file,
        "--is_peerloss": is_peer,
        "--alpha": 0.0,
        "--alpha_scheduler": 'seg',
        "--alpha_list": alpha_list,
        "--milestones": milestones,
        "--gpu_idx": gpu_idx,
        "--chosen_classes": chosen_classes,
        "--sample_weight_path": sample_weight_path,
        "--beta_path": None,
        }, 
        json_path = json_path,
        outputfile= outfile
    )
    Run.train()