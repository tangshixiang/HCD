[Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.0-yellow.svg)

# Online Pseudo Label Generation by Hierarchical Cluster Dynamics for Adaptive Person Re-identification

This repository provides testing code and models of #2775.

## Requirements

### Installation
```shell
python setup.py install
```
### Prepare Datasets

Download the person datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565).
Then unzip them under the directory like

```
HCD/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```
## Evaluation

### Unsupervised Domain Adaptation

To evaluate the model on the target-domain dataset, run:
```shell
CUDA_VISIBLE_DEVICES=0 python test.py --dsbn -d $DATASET --resume $PATH_MODEL
```
*Example:* DukeMTMC-reID -> Market-1501
```shell
CUDA_VISIBLE_DEVICES=0 python test.py --dsbn -d market1501 --resume uda_duke2market.pth.tar
```
### Unsupervised Learning
To evaluate the model, run:
```shell
CUDA_VISIBLE_DEVICES=0 python test.py -d $DATASET --resume $PATH
```
*Example:* DukeMTMC-reID
```shell
CUDA_VISIBLE_DEVICES=0 python test.py -d dukemtmc --resume usl_duke.pth.tar
```
## Trained Models
You can download models in the paper from [Google Drive](https://drive.google.com/drive/folders/1Ykz0n7n8aOPo4YkD0uanf5R9FWE8tUua?usp=sharing).
