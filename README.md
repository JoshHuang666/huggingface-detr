# HuggingFace DETR

HuggingFace DETR Training, Evaluation, Inferencing Guide

## Clone repo 

```
git clone git@github.com:ARG-NCTU/huggingface-detr.git
``` 

## Enter the repo

```bash
cd huggingface-detr
```

## DGX Server Environment

Download Boat Dataset ![Link](http://gofile.me/773h8/UwcuiA7MG) and put them under huggingface-detr directory.

Docker Build

```bash
source build.sh
```

Convert to SQSH file

```bash
source docker2sqsh.sh 
```

Run on DGX Server (Replace your SQSH file path)

```bash
srun -N 1 -p eys3d --mpi=pmix --gres=gpu:8 --ntasks-per-node 8 --container-image dgx_gpu.sqsh --container-writable --pty /bin/bash
```

## PC Environment

Run on PC

```bash
source docker_run.sh
```

## Training, Evaluation, Inferencing

Log in huggingface

```bash
huggingface-cli login
```

Enter the repo

```bash
cd huggingface-detr/
```

Training

```bash
python3 train_detr_boat.py
```

Evaluation

```bash
python3 eval_detr_boat.py
```

Inferencing

```bash
python3 inference_detr_boat.py
```
