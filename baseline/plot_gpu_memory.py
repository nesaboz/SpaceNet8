import sys
import os
#BASELINE = os.path.join(os.path.dirname(os.path.dirname(__file__)),'baseline')
#print(BASELINE)
#sys.path.append(os.path.join(BASELINE))
from cache_train import RunCache, RunConfig
import matplotlib.pyplot as plt
import numpy as np

CACHE_BASE_DIR="/tmp/share/runs/adrs/all-experiments"
cache = RunCache(CACHE_BASE_DIR)

def get_peak_meory(run_config):
    metrics = cache.get_run_metrics(run_config)
    key = 'flood training' if run_config.flood else 'foundation training'
    return metrics[key]['peak_memory']/(1 << 30)

def get_batch_size_peak_memory(model_name, batch_size, flood):
    return get_peak_meory(RunConfig(
        train_csv="/tmp/share/data/spacenet8/adrs-small-train.csv",
        val_csv="/tmp/share/data/spacenet8/adrs-small-train.csv",
        model_name=model_name,
        from_pretrained=True,
        lr=0.0001,
        batch_size=batch_size,
        n_epochs=1,
        flood = flood
    ))

def get_peak_memory_range(model_name, max_batch_size, flood):
    return [
        get_batch_size_peak_memory(model_name, batch_size, flood)
        for batch_size in range(1, max_batch_size + 1)
    ]

def plot_series(series, title, filename):
    fig, ax = plt.subplots()
    for model_name, peak_memories in series.items():
        ax.plot(np.arange(1, len(peak_memories) + 1), peak_memories,
        label=model_name, marker='x')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Peak GPU Memory (GiB)')
    ax.set_title(title)
    ax.legend()

    fig.savefig(os.path.join('baseline', 'results', filename))

def plot():
    foundation_series = {
        model_name:get_peak_memory_range(model_name, max_batch_size, flood=False)
        for model_name, max_batch_size in [
            ("resnet34", 8),
            #("resnet50", 6),
            #("resnet101", 4),
            ("segformer_b0", 7),
            #("segformer_b1", 6),
            #("segformer_b2", 3)
        ]
    }
    plot_series(foundation_series, 'Foundation Network Training Peak Memory', 'peak_gpu_memory_foundation_training.png')
    flood_series = {
        model_name:get_peak_memory_range(model_name, max_batch_size, flood=True)
        for model_name, max_batch_size in [
            ("resnet34_siamese", 6),
            ("segformer_b0_siamese", 4),
            # ("segformer_b1_siamese", 4),
            # ("segformer_b2_siamese", 1),
        ]
    }
    plot_series(flood_series, 'Flood Network Training Peak Memory', 'peak_gpu_memory_flood_training.png')
    plot_series({**flood_series,  **foundation_series}, 'Peak Training Memory', 'peak_training_gpu_memory.png')

if __name__ == '__main__':
    plot()
