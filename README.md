# HuggingFace DETR

HuggingFace DETR Training, Evaluation, Inferencing Guide

## Clone repo 

```
git clone git@github.com:ARG-NCTU/huggingface-detr.git
``` 

## Enter the repo

```bash
cd ~/huggingface-detr/
```

## Enter Docker Environment

For first terminal:

```bash
source Docker/ros1-gpu/run.sh
```

More terminal:

```bash
source Docker/ros1-gpu/join.sh
```

In addition, you can convert docker image to SQSH file (not necessary)

```bash
source docker2sqsh.sh 
```

## Prepare Dataset

Download HuggingFace dataset:

```bash
huggingface-cli login
huggingface-cli download ARG-NCTU/Boat_dataset_2024 data --repo-type dataset --local-dir ~/huggingface-detr
```

Unzip images:

```bash
unzip ~/huggingface-detr/data/images.zip -d ~/huggingface-detr/
```

## Training, Evaluation, Inferencing

Log in huggingface

```bash
huggingface-cli login
```

Enter the repo

```bash
cd ~/huggingface-detr/
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

## Build ROS1 Workspace

Enter the repo

```bash
cd ~/huggingface-detr/
```

Setup ROS

```bash
source environment_ros1.sh 
```

Clean catkin ws

```bash
source clean_ros1_all.sh
```

Build catkin ws

```bash
source build_ros1_all.sh
```

## ROS1 Inference

Setup ROS

```bash
source environment_ros1.sh 
```

Run DETR

```bash
roslaunch detr_inference detr_inference.launch 
```
