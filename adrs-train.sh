#!/bin/bash

REPO_DIR="/tmp/share/repos/adrian/SpaceNet8/"
SAVE_DIR="/tmp/share/runs/adrs/$(date +%Y-%m-%d)"
mkdir -p $SAVE_DIR
python $REPO_DIR/baseline/train_foundation_features.py \
	--train_csv /tmp/share/runs/sn8_data_train.csv \
	--val_csv /tmp/share/runs/sn8_data_val.csv \
	--save_dir $SAVE_DIR \
	--model_name resnet34 \
	--lr 0.0001 \
	--batch_size 4 \
	--n_epochs 1 \
	--gpu 0
