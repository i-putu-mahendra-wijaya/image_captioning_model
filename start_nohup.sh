#!/bin/bash

export TF_FORCE_GPU_ALLOW_GROWTH=true
nohup python3 create_image_captioning_model.py > nohup/create_image_captioning_model.log 2>&1 & echo $! > nohup/create_image_captioning_model.pid

