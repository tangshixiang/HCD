[Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.0-yellow.svg)

# Online Pseudo Label Generation by Hierarchical Cluster Dynamics for Adaptive Person Re-identification
This repo is an official implementation of Online Pseudo Label Generation by Hierarchical Cluster Dynamics for Adaptive Person Re-identification.
This repo is largely based on SpCL. Many thanks to Yixiao Ge. https://github.com/yxgeee/SpCL/tree/master/spcl

## Requirements
Please make ensure the install the dependencies from SpCL.

### Installation
```shell
python setup.py install
```
### Testing the pretrained models
Please refer to https://github.com/AnomoyousCodeReleaser/HCD

## Training the model
```
sh demo.sh
```

## Trained Models
You can download models in the paper from [Google Drive](https://drive.google.com/drive/folders/1Ykz0n7n8aOPo4YkD0uanf5R9FWE8tUua?usp=sharing).

## Citation
```
@inproceedings{zheng2021online,
  title={Online Pseudo Label Generation by Hierarchical Cluster Dynamics for Adaptive Person Re-Identification},
  author={Zheng, Yi and Tang, Shixiang and Teng, Guolong and Ge, Yixiao and Liu, Kaijian and Qin, Jing and Qi, Donglian and Chen, Dapeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8371--8381},
  year={2021}
}
```
