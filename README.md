## Preliminary code for reviewers only


## Prerequisites

* Python 2.7
* Pytorch 0.4
* CUDA 9.0

## Installation the environment

Please first refer to [MattNet](https://github.com/insomnia94/MAttNet) to prepare related data. 

## Pre-trained models
All pre-trained models and related data can be downloaded [here](https://drive.google.com/drive/folders/12HAUdAYNnz6ubiwywcrOEKrVCLiOPiFV). 

Apart from the triads generated by the parsing mechasnism, we also provide the human-annotated triads for all queries for future works, whihc can be downloaded [here](https://drive.google.com/drive/folders/1G3V0NaHnit7omephox_sXTUcedyoa_16?usp=sharing).

## Generate the Triads
```bash
python ./tools/prepro_rel.py --dataset refcoco --splitBy unc
```

## Training

```bash
python ./tools/train.py --dataset refcoco --splitBy unc --exp_id 1
```

## Evaluation

```bash
python ./eval.py --dataset refcoco --splitBy unc --split val --id 1
```

