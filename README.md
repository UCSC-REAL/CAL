# A Second-Order Approach to Learning with Instance-Dependent Label Noise
This code is a PyTorch implementation of the paper "[A Second-Order Approach to Learning with Instance-Dependent Label Noise](https://arxiv.org/abs/2012.11854)" accepted by CVPR 2021 as oral presentation.


## Prerequisites
Python 3.6.9

PyTorch 1.2.0

Torchvision 0.5.0


## Instructions
**Run the code:**
```python
python run_exptPRLD_C10_CAL.py
```
**Construct $\hat D$:**

Modify Lines 27-34 of *run_exptPRLD_C10_CAL.py* as: 
```python
#-------------- customized parameters --------------#
noise_rate = 0.6 # noise rates = 0.2, 0.4, 0.6

lossfunc = "crossentropy"  # use this lossfunc for constructing D
# lossfunc = "crossentropy_CAL" # use this lossfunc for CAL

gpu_idx = "0"   # Choose one GPU index
#---------------------------------------------------#
```


 **Train CAL:**
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
@article{zhu2020second,
  title={A Second-Order Approach to Learning with Instance-Dependent Label Noise},
  author={Zhu, Zhaowei and Liu, Tongliang and Liu, Yang},
  journal={arXiv preprint arXiv:2012.11854},
  year={2020}
}
```

