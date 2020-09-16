## Preliminary code for reviewers only


## Prerequisites

* Python 2.7
* Pytorch 0.4
* CUDA 9.0

## Installation the environment

Please first refer to [MattNet](https://github.com/insomnia94/MAttNet) to prepare related data.

## Pre-trained models
All pre-trained models and related data can be downloaded [here](https://drive.google.com/drive/folders/12HAUdAYNnz6ubiwywcrOEKrVCLiOPiFV).

## Generate the Triads
```bash
python ./tools/prepro_rel.py
```

## Training

```bash
python ./tools/train.py
```

## Evaluation

```bash
python ./eval.py 
```

