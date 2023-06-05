#!/bin/bash

set +x

REPO_DIR="/tmp/share/repos/adrian/SpaceNet8/"
FIG_DIR="/tmp/share/runs/adrs/figs-$(date +%Y-%m-%d-%H%M%S)"
CONFIG=$REPO_DIR/baseline/postprocessing/adrs-config.json

mkdir -p $FIG_DIR

python $REPO_DIR/baseline/postprocessing/create_plots.py \
	--config $CONFIG \
	--fig_dir $FIG_DIR

echo See $FIG_DIR for results
