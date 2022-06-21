# SOTA implementation of Differentially Private SGD (DP-SGD)

This code will implement the algorithm in ["Unlocking High-Accuracy Differentially Private Image Classification through Scale"](https://arxiv.org/abs/2204.13650). For now, this code does not implement Augmentation multiplicity and Parameter averageing (see Table 2 in the paper).



# Environment
This code is tested on Linux system with CUDA version 11.0

Please first install the following packages:

```
python
numpy
torch
opacus
```
# Example commands

An example command that trains a WRN40-4 model on the CIFAR-10 dataset (test accuracy ~= 72% under (8,1e-5)-DP):

```
    python main.py --private --batchsize 500 --accmu 10 --n_epoch 300 --lr 4 --momentum 0 --clip 1 --weight_decay 0 --eps 8 --delta 1e-5 
```
