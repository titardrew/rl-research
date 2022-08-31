#!/bin/bash

python train_ms_e2c.py \
    --env=planar \
    --propor=0.75 \
    --batch_size=128 \
    --lr=0.0001 \
    --lam=0.25 \
    --num_iter=5000 \
    --iter_save=100 \
    --log_dir=logs/new \
    --seed=42 \
    --n_steps=$1
