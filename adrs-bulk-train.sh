#!/bin/bash

set +x

REPO_DIR="/tmp/share/repos/adrian/SpaceNet8/"
SAVE_DIR="/tmp/share/runs/adrs/bulk-run-$(date +%Y-%m-%d-%H%M%S)"

# Dateset
#########################################################
# # Full dataset
# TRAIN_CSV="/tmp/share/data/spacenet8/sn8_data_train.csv"
# VAL_CSV="/tmp/share/data/spacenet8/sn8_data_val.csv"

# Partial dataset to speed up test runs
TRAIN_CSV="/tmp/share/data/spacenet8/adrs-small-train.csv"
VAL_CSV="/tmp/share/data/spacenet8/adrs-small-val.csv"

mkdir -p $SAVE_DIR

python $REPO_DIR/baseline/bulk_train.py \
	--save_dir $SAVE_DIR \
	--train_csv $TRAIN_CSV \
	--val_csv $VAL_CSV \
    --foundation_model_names resnet34 segformer_b0 resnet34 segformer_b0 \
	--foundation_model_from_pretrained true true false false \
	--foundation_lr 0.0001 \
	--foundation_batch_size 2 \
	--foundation_n_epochs 1 \
    --flood_model_names resnet34_siamese segformer_b0_siamese resnet34_siamese segformer_b0_siamese \
	--flood_model_from_pretrained true true false false \
	--flood_lr 0.0001 \
	--flood_batch_size 2 \
	--flood_n_epochs 1 \
	--gpu 0

echo See $SAVE_DIR for results
