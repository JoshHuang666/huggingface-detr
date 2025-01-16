#!/usr/bin/env bash

docker run \
    -it \
    --rm \
    -w "/home/arg" \
    --user "root:root" \
    argnctu/dgx:gpu \
