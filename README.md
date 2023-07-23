# Modified-SCINet

## Background

This repository provide modified SCINet model to predict stock price and implement based on PyTorch Framework.

## How to use
```
python main.py
```
## Configuration Setup

`seq_len` is the number of input to forecast output\
`pred_len` is the number of step size to forecast the step of output\
`in_dim` is the number of dimension of input (feature inputs)\
`kernel` is the kernel of convolution 1D, should be odd value (3,5,7,...)