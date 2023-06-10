"""
the goal is to run several models, and then compare the time per epoch vs the number of parameters
n_params nad time per epoch will be stored in metrics.json
"""

import sys
import os
BASELINE = '/tmp/share/repos/naijing/SpaceNet8/baseline/'
print(BASELINE)
sys.path.append(BASELINE)
# print(sys.path)
from pathlib import Path
from datetime import datetime
from end2end import run
from utils.log import load_from_json
import os
import glob
from matplotlib import pyplot as plt
import numpy as np


train_csv="/tmp/share/data/spacenet8/sn8_data_train.csv"
val_csv="/tmp/share/data/spacenet8/sn8_data_val.csv"

# train_csv="/tmp/share/data/spacenet8/adrs-small-train.csv"
# val_csv="/tmp/share/data/spacenet8/adrs-small-val.csv"
run_root = Path('/tmp/share/runs/spacenet8/naijing')

def run_experiment():
    """
    Howe long it takes to train an epoch.
    """
    now = datetime.now() 
    folder = os.path.join(run_root, now.strftime("%Y-%m-%d-%H-%M"))

    # flood_model_name = 'dense_161_siamese'
    flood_model_name = 'effunet_b4_siamese'
    # model_name = 'dense_161'
    model_name = 'effunet_b4'

    run(
        save_dir=os.path.join(folder, model_name),
        train_csv=train_csv,
        val_csv=val_csv,
        foundation_model_name=model_name,
        foundation_lr=0.0001,
        foundation_batch_size=2,
        foundation_n_epochs=10,
        flood_model_name=flood_model_name,
        flood_lr=0.0001,
        flood_batch_size=1,
        flood_n_epochs= 10
    )
         
    return folder
    

if __name__ == '__main__':
    folder = run_experiment()
        