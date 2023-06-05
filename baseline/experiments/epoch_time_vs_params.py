"""_summary_
the goal is to run several models, for 1 epoch each, and then compare the time per epoch vs the number of parameters
time per epoch will be stored as epoch_duration in metrics
n_params will be stored in params.json (double check both of these)
let's run each model for 3 epoch, and average the time per epoch
let's first run segformer_b0, then segformer_b1, then resnet34.

params.json and metrics.json will be stored per model in each folder, 
let's just add a note to the folder name to indicate which model it is and it's purpose
like, segformer_b0_3epoch, segformer_b1_3epoch, resnet34_3epoch

1) double check params.json and metrics.json have the two variables
2) install fabric and test it out 

"""

from pathlib import Path
from datetime import datetime
from ..end2end import run
import os

# train_csv="/tmp/share/data/spacenet8/sn8_data_train.csv"
# val_csv="/tmp/share/data/spacenet8/sn8_data_val.csv"

train_csv="/tmp/share/data/spacenet8/adrs-small-train.csv"
val_csv="/tmp/share/data/spacenet8/adrs-small-val.csv"

run_root = Path('/tmp/share/runs/spacenet8/nenad')
now = datetime.now() 

for model_name in ['segformer_b0', 'segformer_b1', 'resnet34']:
    try:
        run(
            save_dir=os.path.join(run_root, model_name + '_3epochs_' + now.strftime("%d-%m-%Y-%H-%M")),
            train_csv=train_csv,
            val_csv=val_csv,
            foundation_model_name=model_name,
            foundation_lr=0.0001,
            foundation_batch_size=4,
            foundation_n_epochs=2,
            foundation_checkpoint=None,
            foundation_model_args={},
            foundation_kwargs={}
        )
    except:
        print(f'failed to run {model_name}')
        continue
    
for model_name in ['resnet50', 'seresnext50', 'unet']:
    try:
        run(
            save_dir=os.path.join(run_root, model_name + '_3epochs_' + now.strftime("%d-%m-%Y-%H-%M")),
            train_csv=train_csv,
            val_csv=val_csv,
            foundation_model_name=model_name,
            foundation_lr=0.0001,
            foundation_batch_size=4,
            foundation_n_epochs=2,
            foundation_checkpoint=None,
            foundation_model_args={},
            foundation_kwargs={}
        )
    except:
        print(f'failed to run {model_name}')
        continue
    
    
