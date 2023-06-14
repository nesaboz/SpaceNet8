#!/bin/bash

set +x

REPO_DIR="/tmp/share/repos/adrian/SpaceNet8/"
SAVE_DIR="/tmp/share/runs/adrs/all-experiments"

DATA_CSV="/tmp/share/data/spacenet8/adrs-small-train.csv"
#VAL_CSV="/tmp/share/data/spacenet8/adrs-small-val.csv"

echo "Training P1 models ..."
python $REPO_DIR/baseline/cache_train.py \
	--save_dir $SAVE_DIR \
	--train_csv $DATA_CSV \
	--val_csv $DATA_CSV \
    --foundation_model_names \
	  resnet34 \
	  resnet34 \
	  resnet34 \
	  resnet34 \
	  resnet34 \
	  resnet34 \
	  resnet34 \
	  resnet34 \
	  segformer_b0 \
	  segformer_b0 \
	  segformer_b0 \
	  segformer_b0 \
	  segformer_b0 \
	  segformer_b0 \
	  segformer_b0 \
	  segformer_b0 \
	--foundation_model_from_pretrained true \
	--foundation_lr 0.0001 \
	--foundation_batch_size \
		1 \
		2 \
		3 \
		4 \
		5 \
		6 \
		7 \
		8 \
		1 \
		2 \
		3 \
		4 \
		5 \
		6 \
		7 \
		8 \
	--foundation_n_epochs 1 \
	--gpu 0

python $REPO_DIR/baseline/cache_train.py \
	--save_dir $SAVE_DIR \
	--train_csv $DATA_CSV \
	--val_csv $DATA_CSV \
    --flood_model_names \
	  resnet34_siamese \
	  resnet34_siamese \
	  resnet34_siamese \
	  resnet34_siamese \
	  resnet34_siamese \
	  resnet34_siamese \
	  segformer_b0_siamese \
	  segformer_b0_siamese \
	  segformer_b0_siamese \
	  segformer_b0_siamese \
	--flood_model_from_pretrained true \
	--flood_lr 0.0001 \
	--flood_batch_size \
		1 \
		2 \
		3 \
		4 \
		5 \
		6 \
		1 \
		2 \
		3 \
		4 \
	--flood_n_epochs 1 \
	--gpu 0

exit 1
echo "Training P2 models ..."
python $REPO_DIR/baseline/cache_train.py \
	--save_dir $SAVE_DIR \
	--train_csv $DATA_CSV \
	--val_csv $DATA_CSV \
    --foundation_model_names \
	  resnet50 \
	  resnet50 \
	  resnet50 \
	  resnet50 \
	  resnet50 \
	  resnet50 \
	  segformer_b1 \
	  segformer_b1 \
	  segformer_b1 \
	  segformer_b1 \
	  segformer_b1 \
	  segformer_b1 \
	  resnet101 \
	  resnet101 \
	  resnet101 \
	  resnet101 \
	  segformer_b3 \
	  segformer_b3 \
	  segformer_b3 \
	--foundation_model_from_pretrained true \
	--foundation_lr 0.0001 \
	--foundation_batch_size \
		1 \
		2 \
		3 \
		4 \
		5 \
		6 \
		1 \
		2 \
		3 \
		4 \
		5 \
		6 \
		1 \
		2 \
		3 \
		4 \
		1 \
		2 \
		3 \
	--foundation_n_epochs 1 \
	--gpu 0

python $REPO_DIR/baseline/cache_train.py \
	--save_dir $SAVE_DIR \
	--train_csv $DATA_CSV \
	--val_csv $DATA_CSV \
    --flood_model_names \
	  segformer_b1_siamese \
	  segformer_b1_siamese \
	  segformer_b1_siamese \
	  segformer_b1_siamese \
	  segformer_b2_siamese \
	--flood_model_from_pretrained true \
	--flood_lr 0.0001 \
	--flood_batch_size \
		1 \
		2 \
		3 \
		4 \
		1 \
	--flood_n_epochs 1 \
	--gpu 0
