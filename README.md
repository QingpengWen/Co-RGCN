# **Co-RGCN: A Bi-path GCN-based Co-Regression model for Multi-intent Detection and Slot Filling**

This repository contains the official `PyTorch` implementation of the paper: 

**Co-RGCN: A Bi-path GCN-based Co-Regression model for Multi-intent Detection and Slot Filling**.

[Qingpeng Wen](mailto:wqp@mail2.gdut.edu.cn), [Bi Zeng](mailto:zb9215@gdut.edu.cn), [Pengfei Wei](mailto:wpf@gdut.edu.cn).

In the following, we will guide you how to use this repository step by step.

## Architecture

<img src="Figures\fig1.png">

## Preparation

Our code is based on PyTorch 1.12 Required python packages:

-   numpy==1.19.1
-   tqdm==4.50.0
-   pytorch==1.12.1
-   python==3.8.5
-   cudatoolkit==10.0.130
-   fitlog==0.7.1
-   ordered-set==4.0.2
-   scipy==1.10.1
-   transformers==4.27.1
-   spacy==3.5.0

We highly suggest you using [Anaconda](https://www.anaconda.com/) to manage your python environment.

## How to run it
The script **train.py** acts as a main function to the project, you can run the experiments by the following commands.
```Shell
# MixATIS dataset (ON GeForce RTX3090)
python train.py -g -bs=16 -dd=./data/MixATIS -sd=./save/MixATIS -lod=./log/MixSNIPS

# MixSNIPS dataset (ON GeForce RTX3090)
python train.py -g -bs=16 -dd=./data/MixSNIPS  -sd=./save/MixSNIPS -lod=./log/MixATIS
```

If you have any question, please issue the project or email [me](mailto:wqp@mail2.gdut.edu.cn)  and we will reply you soon.

