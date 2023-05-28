#!/bin/bash

set +x

REPO_DIR="/tmp/share/repos/adrian/SpaceNet8/"
SAVE_DIR="/tmp/share/runs/adrs/train-flood-$(date +%Y-%m-%d-%H%M%S)"
MODEL_NAME=resnet34_siamese
#MODEL_NAME=segformer_b0_siamese

# Full dataset
# TRAIN_CSV="/tmp/share/runs/sn8_data_train.csv"
# VAL_CSV="/tmp/share/runs/sn8_data_val.csv"

# Partial dataset to speed up test runs
TRAIN_CSV="/tmp/share/data/spacenet8/adrs-small-train.csv"
VAL_CSV="/tmp/share/data/spacenet8/adrs-small-val.csv"

mkdir -p $SAVE_DIR

python $REPO_DIR/baseline/train_flood.py \
	--train_csv $TRAIN_CSV \
	--val_csv $VAL_CSV \
	--save_dir $SAVE_DIR \
	--model_name $MODEL_NAME \
	--lr 0.0001 \
	--batch_size 2 \
	--n_epochs 1 \
	--gpu 0
