"""_summary_
the goal is to run Segformer with different segmentation layers, compare IoUs.
segformer_b0_1x1_conv, segformer_b0_double_conv, and existing segformer_b0
"""

import sys
import torch
import os
BASELINE = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASELINE)
from pathlib import Path
from datetime import datetime
from end2end import run
from utils.log import load_from_json
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import models.other.segformer as segformer
from utils.utils import freeze_model, unfreeze_model, count_parameters


train_csv="/tmp/share/data/spacenet8/sn8_data_train.csv"
val_csv="/tmp/share/data/spacenet8/sn8_data_val.csv"

# train_csv="/tmp/share/data/spacenet8/adrs-small-train.csv"
# val_csv="/tmp/share/data/spacenet8/adrs-small-val.csv"
run_root = Path('/tmp/share/runs/spacenet8/nenad')

def run_experiment():
    """
    Run training different types of heads for segformer.
    """
    now = datetime.now() 
    tag = '_head'
    folder = os.path.join(run_root, now.strftime("%Y-%m-%d-%H-%M") + tag)
        
    for model_name in ['segformer_b0_1x1_conv', 'segformer_b0_double_conv']:
        run(
            save_dir=os.path.join(folder, model_name + tag),
            train_csv=train_csv,
            val_csv=val_csv,
            foundation_model_name=model_name,
            foundation_lr=0.0001,
            foundation_batch_size=4,
            foundation_n_epochs=10,
            foundation_checkpoint=None,
            foundation_model_args={},
            foundation_kwargs={}
        )


if __name__ == '__main__':
    run_experiment()

        