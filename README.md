# **Co-RGCN: A Bi-path GCN-based Co-Regression model for Multi-intent Detection and Slot Filling**

This repository contains the official `PyTorch` implementation of the paper in the 32nd International Conference on Artificial Neural Networks, which collected at Artificial Neural Networks and Machine Learning (***ICANN2023***): 

**[Co-RGCN: A Bi-path GCN-based Co-Regression model for Multi-intent Detection and Slot Filling](https://link.springer.com/chapter/10.1007/978-3-031-44216-2_26)**.

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

## Acknowledgement

This work was supported in part by the National Science Foundation of China under Grant 62172111, in part by the Natural Science Foundation of Guangdong Province under Grant 2019A1515011056, in part by the Key technology project of Shunde District under Grant 2130218003002.

## Cite this paper

@InProceedings{

[10.1007/978-3-031-44216-2_26](https://doi.org/10.1007/978-3-031-44216-2_26),

author="Wen, Qingpeng

and Zeng, Bi

and Wei, Pengfei",

editor="Iliadis, Lazaros

and Papaleonidas, Antonios

and Angelov, Plamen

and Jayne, Chrisina",

title="Co-RGCN: A Bi-path GCN-Based Co-Regression Model for Multi-intent Detection and Slot Filling",

booktitle="Artificial Neural Networks and Machine Learning -- ICANN 2023",

year="2023",

publisher="Springer Nature Switzerland",

address="Cham",

pages="316--327",

isbn="978-3-031-44216-2"

}
