"""
the goal is to run several models, and then compare the time per epoch vs the number of parameters
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
from matplotlib import pyplot as plt
import numpy as np


train_csv="/tmp/share/data/spacenet8/sn8_data_train.csv"
val_csv="/tmp/share/data/spacenet8/sn8_data_val.csv"

# train_csv="/tmp/share/data/spacenet8/adrs-small-train.csv"
# val_csv="/tmp/share/data/spacenet8/adrs-small-val.csv"
run_root = Path('/tmp/share/runs/spacenet8/nenad')

def run_experiment():
    """
    How long it takes to train an epoch.
    """
    now = datetime.now() 
    folder = os.path.join(run_root, now.strftime("%Y-%m-%d-%H-%M"))

    for model_name in ['effunet_b2', 'effunet_b4', 'dense_121', 'dense_161']:  # ['dummy', 'segformer_b0', 'segformer_b1', 'resnet34', 'resnet50']:
        print(f'Runnning {model_name} ...')
        try:
            run(
                save_dir=os.path.join(folder, '_epoch_time_' + model_name),
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
    return folder
    
def plot_epoch_time(folder, label, tag=''):
    """
    Get all the `epochs_time_vs_params` folders, and look into metrics.json for epoch_duration, n_params, and model name
    """
    # get all the folders with the pattern *epochs_time_vs_params* in them
    folders = list(Path(folder).glob(f'*{label}*'))
    tag = '_' + tag if tag else ''
    
    n_params = []
    learnable_n_params = []
    epoch_times = []
    epoch_times_short = []
    model_names = []
    for folder in folders:
        try:
            # read metrics.json in the folder
            metrics = load_from_json(folder/'metrics.json')['foundation training']
            model_name = metrics['model_name']
            learnable_n_param = metrics['learnable_parameter_count']
            n_param = metrics['parameter_count'] / 1e6
            temp = [metrics['epochs'][i]['epoch_duration'] for i in range(len(metrics['epochs']))]
            temp_short = [metrics['epochs'][i]['epoch_duration_shorter'] for i in range(len(metrics['epochs']))]
            epoch_time = np.mean([x/60.0 for x in temp])
            epoch_time_short = np.mean([x/60.0 for x in temp_short])
        except:
            continue
        n_params.append(n_param)
        epoch_times.append(epoch_time)
        epoch_times_short.append(epoch_time_short)
        model_names.append(model_name)
        learnable_n_params.append(learnable_n_param)
        
    def plot_scatter(x, y, labels):
        assert len(x) == len(y) == len(labels), "Input lists must have the same length"

        plt.scatter(x, y)
        for i, label in enumerate(labels):
            plt.annotate(label, (x[i], y[i]+0.1))

        plt.xlabel('Number of parameters (M)')
        
        plt.ylabel('Time per epoch (minutes)')
        plt.title('Forward + Backward pass duration for Foundation models')
        plt.grid(True)
        now = datetime.now() 
        plt.ylim(0, 5)
        plt.xlim(-2, 1.2*max(x))
        plt.plot([-2, 1.2*max(x)], [4.5, 4.5], linestyle='--', color='red')
        plt.annotate('Data transfer time on A6000', (0, 4.5+0.1))

        # plt.axis([0, 1.1*max(x), 0, 1.1*max(y)])
        plt.savefig(os.path.join(BASELINE, f'results/epoch_time{tag}.png'))
        print(f'Created file.')
        plt.show()

    plot_scatter(n_params, epoch_times_short, model_names)

if __name__ == '__main__':
    # folder = run_experiment()
    # epoch duration shorter /tmp/share/runs/spacenet8/nenad/2023-06-07-21-47
    # epoch duration '/tmp/share/runs/spacenet8/nenad/2023-06-05-18-49_epoch_time_vs_param')  # latest run: '/tmp/share/runs/spacenet8/nenad/2023-06-05-18-49_epoch_time_vs_param'
    plot_epoch_time('/tmp/share/runs/spacenet8/nenad/2023-06-07-21-47', tag='shorter')  # 
    plot_epoch_time('/tmp/share/runs/spacenet8/nenad/2023-06-05-18-49_epoch_time_vs_param')  # latest run: '/tmp/share/runs/spacenet8/nenad/2023-06-05-18-49_epoch_time_vs_param')  # 
        