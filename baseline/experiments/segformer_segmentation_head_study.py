"""
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
from matplotlib.transforms import Bbox


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

def plot_segmentation_head():
    """
    We aggregate metrics from 3 runs, look only at foundation loss, plot the three loss graphs (line plot) and IoUs (bar plot)
    """
    run_folder = Path('/tmp/share/runs/spacenet8/nenad/2023-06-06-02-49_head')
    model_names = ['segformer_b0_1x1_conv_head', 'segformer_b0_double_conv_head', 'segformer_b0']
    model_labels = ['1x1 conv', 'double 3x3', '3x3 conv']
    losses = []
    _, ax = plt.subplots(1, 1, figsize=(5, 3))
                            
    colors = ('blue', 'red', 'green')
    iou = []
    for color, model_name, model_label in zip(colors, model_names, model_labels):
        loss = []
        metrics_training = load_from_json(run_folder/model_name/'metrics.json')['foundation training']
        metrics_eval = load_from_json(run_folder/model_name/'metrics.json')['foundation eval']
        
        model_name = metrics_training['model_name']
        epoch_data = metrics_training['epochs']
        n_epochs = len(epoch_data)
        train_tot_loss = [epoch_data[i]['train_tot_loss'] for i in range(n_epochs)]
        val_tot_loss = [epoch_data[i]['val_tot_loss'] for i in range(n_epochs)]
        losses.append(loss)
        ax.plot(range(n_epochs), train_tot_loss, label='train_loss_' + model_label, color=color)
        ax.plot(range(n_epochs), val_tot_loss, '--', label='val_loss_' + model_label, color=color)
        
        iou.append(metrics_eval['metrics_by_class']['building']['iou'])
         
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Segformer head comparison')
    ax.legend(loc="upper right", fontsize="small")    
    
    print(", ".join([f'{b:.3f} ({a})' for a, b in zip(model_labels, iou)]))
    plt.savefig(os.path.join(BASELINE, f'results/segformer_head_comparison.png'),
                dpi=300, 
                bbox_inches=Bbox.from_extents(-0.2, -0.2, 5, 3))
    plt.show()
            

if __name__ == '__main__':
    # run_experiment()
    plot_segmentation_head()

        