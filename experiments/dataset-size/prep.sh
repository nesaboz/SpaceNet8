#!/bin/bash

set +x

REPO_DIR="./"

# Full dataset
# The input file was created from randomly shuffling the entire dataset.
TRAIN_CSV="/tmp/share/data/spacenet8/sn8_data_train.csv"
VAL_CSV="/tmp/share/data/spacenet8/sn8_data_val.csv"

OUTPUT_DIR="$REPO_DIR/experiments/dataset-size"

python $REPO_DIR/experiments/dataset-size/prep.py \
	--train_csv $TRAIN_CSV \
	--val_csv $VAL_CSV \
	--output_dir $OUTPUT_DIR \
	--fractions 0.2 0.4 0.6 0.8 1
