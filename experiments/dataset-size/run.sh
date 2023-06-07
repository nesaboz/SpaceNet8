#!/bin/bash

set +x

REPO_DIR="/tmp/share/repos/adrian/SpaceNet8/"
SAVE_DIR="/tmp/share/runs/adrs/dataset-size-$(date +%Y-%m-%d-%H%M%S)"

TRAIN_CSV="/tmp/share/data/spacenet8/adrs-small-train.csv"
VAL_CSV="/tmp/share/data/spacenet8/adrs-small-val.csv"
EXPERIMENT_DIR="$REPO_DIR/experiments/dataset-size"

FOUNDATION="resnet34"
FLOOD="resnet34_siamese"

mkdir -p $SAVE_DIR

python $REPO_DIR/baseline/bulk_train.py \
	--save_dir $SAVE_DIR \
	--train_csv $EXPERIMENT_DIR/train-0.2.csv \
	            $EXPERIMENT_DIR/train-0.4.csv \
	            $EXPERIMENT_DIR/train-0.6.csv \
	            $EXPERIMENT_DIR/train-0.8.csv \
	            $EXPERIMENT_DIR/train-1.0.csv \
	--val_csv $EXPERIMENT_DIR/val.csv \
    --foundation_model_names $FOUNDATION $FOUNDATION $FOUNDATION $FOUNDATION $FOUNDATION \
	--foundation_model_from_pretrained true \
	--foundation_lr 0.0001 \
	--foundation_batch_size 4 \
	--foundation_n_epochs 1 \
    --flood_model_names $FLOOD $FLOOD $FLOOD $FLOOD $FLOOD \
	--flood_model_from_pretrained true \
	--flood_lr 0.0001 \
	--flood_batch_size 4 \
	--flood_n_epochs 1 \
	--gpu 0

echo See $SAVE_DIR for results
