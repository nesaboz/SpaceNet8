#!/bin/bash

set +x

REPO_DIR="/tmp/share/repos/adrian/SpaceNet8/"
SAVE_DIR="/tmp/share/runs/adrs/all-experiments"

TRAIN_CSV="/tmp/share/data/spacenet8/adrs-small-train.csv"
VAL_CSV="/tmp/share/data/spacenet8/adrs-small-val.csv"

mkdir -p $SAVE_DIR

#FOUNDATION="resnet34"
#FOUNDATION="segformer_b2"
FLOOD="segformer_b2_siamese"
BATCH_SIZE=1

# python $REPO_DIR/baseline/cache_train.py \
# 	--save_dir $SAVE_DIR \
# 	--train_csv $VAL_CSV \
# 	--val_csv $VAL_CSV \
#     --foundation_model_names $FOUNDATION \
# 	--foundation_model_from_pretrained true \
# 	--foundation_lr 0.0001 \
# 	--foundation_batch_size $BATCH_SIZE \
# 	--foundation_n_epochs 1 \
# 	--gpu 0

python $REPO_DIR/baseline/cache_train.py \
	--save_dir $SAVE_DIR \
	--train_csv $VAL_CSV \
	--val_csv $VAL_CSV \
    --flood_model_names $FLOOD \
	--flood_model_from_pretrained true \
	--flood_lr 0.0001 \
	--flood_batch_size $BATCH_SIZE \
	--flood_n_epochs 1 \
	--gpu 0

echo See $SAVE_DIR for results
