# A Second-Order Approach to Learning with Instance-Dependent Label Noise (CVPR'21 oral)
This code is a PyTorch implementation of the paper "[A Second-Order Approach to Learning with Instance-Dependent Label Noise](https://arxiv.org/abs/2012.11854)" accepted by CVPR 2021 as oral presentation.


# Prerequisites
Python 3.6.9

PyTorch 1.4.0

Torchvision 0.5.0


# Instructions
**Run the code:**

CIFAR10:
```python
python run_exptPRLD_C10_CAL.py
```

CIFAR100:
```python
python run_exptPRLD_C100_CAL.py
```

The following changes also apply to CIFAR100.


# Run the code step-by-step
## Step-1: **Construct $\hat D$:**

Modify Lines 27-34 of *run_exptPRLD_C10_CAL.py* as: 
```python
#-------------- customized parameters --------------#
noise_rate = 0.6 # noise rates = 0.2, 0.4, 0.6

lossfunc = "crossentropy"  # use this lossfunc for constructing D
# lossfunc = "crossentropy_CAL" # use this lossfunc for CAL

gpu_idx = "0"   # Choose one GPU index
#---------------------------------------------------#
```

## Step-2: **Train CAL:**
Modify Lines 27-34 of *run_exptPRLD_C10_CAL.py* as: 
```python
#-------------- customized parameters --------------#
noise_rate = 0.6 # noise rates = 0.2, 0.4, 0.6

# lossfunc = "crossentropy"  # use this lossfunc for constructing D
lossfunc = "crossentropy_CAL" # use this lossfunc for CAL

gpu_idx = "0"   # Choose one GPU index
#---------------------------------------------------#
```


## Citation

If you find this code useful, please cite the following paper:

```
@inproceedings{zhu2021second,
  title={A second-order approach to learning with instance-dependent label noise},
  author={Zhu, Zhaowei and Liu, Tongliang and Liu, Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10113--10123},
  year={2021}
}
```
##
Corresponding authors:

Dr. [Zhaowei Zhu](https://users.soe.ucsc.edu/~zhaoweizhu/): zwzhu@ucsc.edu

Prof. [Yang Liu](http://www.yliuu.com/): yangliu@ucsc.edu
