#!/usr/bin/env bash

docker run --gpus all \
    -it \
    --rm \
    -w "/home/arg" \
    --user "root:root" \
    argnctu/huggingface-detr:dgx \
