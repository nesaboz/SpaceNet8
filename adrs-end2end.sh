#!/bin/bash

set +x

REPO_DIR="/tmp/share/repos/adrian/SpaceNet8/"
SAVE_DIR="/tmp/share/runs/adrs/run-$(date +%Y-%m-%d-%H%M%S)"

# Dateset
#########################################################
# # Full dataset
# TRAIN_CSV="/tmp/share/data/spacenet8/sn8_data_train.csv"
# VAL_CSV="/tmp/share/data/spacenet8/sn8_data_val.csv"

# Partial dataset to speed up test runs
TRAIN_CSV="/tmp/share/data/spacenet8/adrs-small-train.csv"
VAL_CSV="/tmp/share/data/spacenet8/adrs-small-val.csv"

# Architectures
#########################################################
#FOUNDATION_MODEL_NAME=resnet34
FOUNDATION_MODEL_NAME=segformer_b0
#FLOOD_MODEL_NAME=resnet34_siamese
FLOOD_MODEL_NAME=segformer_b0_siamese

mkdir -p $SAVE_DIR

python $REPO_DIR/baseline/end2end.py \
	--save_dir $SAVE_DIR \
	--train_csv $TRAIN_CSV \
	--val_csv $VAL_CSV \
	--foundation_model_name $FOUNDATION_MODEL_NAME \
	--foundation_lr 0.0001 \
	--foundation_batch_size 2 \
	--foundation_n_epochs 1 \
	--flood_model_name $FLOOD_MODEL_NAME \
	--flood_lr 0.0001 \
	--flood_batch_size 2 \
	--flood_n_epochs 1 \
	--gpu 0
