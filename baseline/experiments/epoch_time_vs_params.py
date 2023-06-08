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
    plt.ylim(0, 1.1*max(y))
    plt.xlim(-2, 1.2*max(x))
    plt.plot([-2, 1.2*max(x)], [4.5, 4.5], linestyle='--', color='red')
    plt.annotate('Data transfer per epoch (A6000)', (0, 4.5+0.1))

    # plt.axis([0, 1.1*max(x), 0, 1.1*max(y)])
    plt.savefig(os.path.join(BASELINE, f'results/epoch_time.png'))
    print(f'Created file.')
    plt.show()
    
def plot_epoch_time():
    """
    Get all the `epochs_time_vs_params` folders, and look into metrics.json for epoch_duration, n_params, and model name
    """
    
    run_paths = ['/tmp/share/runs/spacenet8/nenad/2023-06-07-21-47_epoch_time/_epoch_time_dummy',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-07-21-47_epoch_time/_epoch_time_resnet34',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-07-21-47_epoch_time/_epoch_time_resnet50',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-07-21-47_epoch_time/_epoch_time_segformer_b0',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-07-21-47_epoch_time/_epoch_time_segformer_b1',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-08-02-20_segformer_b2',
                 ]
    model_names = ['dummy', 'resnet34', 'resnet50', 'segformer_b0', 'segformer_b1',  'segformer_b2'] #  'effunet_b2', 'effunet_b4']
    
    n_params = []
    learnable_n_params = []
    epoch_times = []
    epoch_times_short = []

    for run_path, model_name in zip(run_paths, model_names):
        metrics = load_from_json(os.path.join(run_path, 'metrics.json'))['foundation training']
        model_name = metrics['model_name']
        learnable_n_param = metrics['learnable_parameter_count']
        n_param = metrics['parameter_count'] / 1e6
        temp = [metrics['epochs'][i]['epoch_duration'] for i in range(len(metrics['epochs']))]
        temp_short = [metrics['epochs'][i]['epoch_duration_shorter'] for i in range(len(metrics['epochs']))]
        epoch_time = np.mean([x/60.0 for x in temp])
        epoch_time_short = np.mean([x/60.0 for x in temp_short])
    
        n_params.append(n_param)
        epoch_times.append(epoch_time)
        epoch_times_short.append(epoch_time_short)
        learnable_n_params.append(learnable_n_param)
        
    plot_scatter(n_params, epoch_times_short, model_names)

if __name__ == '__main__':
    # epoch duration shorter /tmp/share/runs/spacenet8/nenad/2023-06-07-21-47
    # epoch duration '/tmp/share/runs/spacenet8/nenad/2023-06-05-18-49_epoch_time_vs_param')  # latest run: '/tmp/share/runs/spacenet8/nenad/2023-06-05-18-49_epoch_time_vs_param'
    # plot_epoch_time('/tmp/share/runs/spacenet8/nenad/2023-06-07-21-47', tag='shorter')  # 
    # plot_epoch_time('/tmp/share/runs/spacenet8/nenad/2023-06-05-18-49_epoch_time_vs_param')  # latest run: '/tmp/share/runs/spacenet8/nenad/2023-06-05-18-49_epoch_time_vs_param')  # 
    plot_epoch_time()
        