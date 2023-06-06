"""
the goal is to run several models, for 3 epoch each, and then compare the time per epoch vs the number of parameters
n_params nad time per epoch will be stored in metrics.json
"""

import sys
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


train_csv="/tmp/share/data/spacenet8/sn8_data_train.csv"
val_csv="/tmp/share/data/spacenet8/sn8_data_val.csv"

# train_csv="/tmp/share/data/spacenet8/adrs-small-train.csv"
# val_csv="/tmp/share/data/spacenet8/adrs-small-val.csv"
run_root = Path('/tmp/share/runs/spacenet8/nenad')

def run_experiment():
    """
    Run training for 3 epoch and see how long it takes per epoch, plot this time 
    vs number of parameters.
    """
    now = datetime.now() 

    for model_name in ['segformer_b0', 'segformer_b1', 'resnet34', 'resnet50', 'seresnext50', 'unet']:
        print(f'Runnning {model_name} ...')
        try:
            run(
                save_dir=os.path.join(run_root, now.strftime("%d-%m-%Y-%H-%M"), '_3epochs_' + model_name),
                train_csv=train_csv,
                val_csv=val_csv,
                foundation_model_name=model_name,
                foundation_lr=0.0001,
                foundation_batch_size=4,
                foundation_n_epochs=1,
                include_eval = False
            )
        except:
            print(f'failed to run {model_name}')
            continue    
    
def plot_experiment():
    """
    Get all the 3epoch folders, and look into metrics.json for epoch_duration, n_params, and model name, and average over 3 epochs
    """
    # get all the folders with the pattern *3epochs* in them
    folders = list(Path('/tmp/share/runs/spacenet8/nenad/05-06-2023-18-49').glob('*3epochs*'))

    n_params = []
    learnable_n_params = []
    epoch_times = []
    epoch_times_std = []
    model_names = []
    for folder in folders:
        try:
            # read metrics.json in the folder
            metrics = load_from_json(folder/'metrics.json')['foundation training']
            model_name = metrics['model_name']
            learnable_n_param = metrics['learnable_parameter_count']
            n_param = metrics['parameter_count']
            epoch_times_temp = [metrics['epochs'][i]['epoch_duration'] for i in range(len(metrics['epochs']))]
            epoch_time = np.mean(epoch_times_temp)
            epoch_time_std = np.std(epoch_times_temp)
        except:
            continue
        n_params.append(n_param)
        epoch_times.append(epoch_time)
        epoch_times_std.append(epoch_time_std)
        model_names.append(model_name)
        learnable_n_params.append(learnable_n_param)
        

    def plot_scatter(x, y, labels):
        assert len(x) == len(y) == len(labels), "Input lists must have the same length"

        plt.scatter(x, y)
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]))

        plt.xlabel('Number of parameters')
        plt.ylabel('epoch duration (s)')
        plt.title('Epoch duration vs number of learnable parameters')
        plt.grid(True)
        plt.savefig(os.path.join(BASELINE, f'results/epoch_time_vs_params.png'))
        plt.show()

    plot_scatter(n_params, epoch_times, model_names)

if __name__ == '__main__':
    # run_experiment()
    plot_experiment()
        
