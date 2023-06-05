import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        required=True)
    parser.add_argument("--fig_dir",
                         type=str,
                         required=True)
    return parser.parse_args()

def load_metric_files(paths):
    return [load_json(path) for path in paths]

def get_peak_gpu_memory(runs_metrics, run_type, model_name, batch_size=None):
    for run_metrics in runs_metrics:
        run_type_metrics = run_metrics.get(run_type)
        if not run_type_metrics:
            continue
        if run_type_metrics.get('model_name') != model_name:
            continue
        if batch_size is not None and run_type_metrics['batch_size'] != batch_size:
            continue
        return run_type_metrics['peak_memory']/(1<<30)


def create_grouped_bar_chart(labels, values_by_subgroup, title, ylabel):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    x = np.arange(len(labels))
    width = 1.0/(len(values_by_subgroup) + 1)
    multiplier = 0
    print(title)

    fig, ax = plt.subplots()
    for subgroup, values in values_by_subgroup.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=subgroup)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width, labels)
    ax.legend(loc='upper left')#, ncols=len(values_by_subgroup))
    return fig

def create_bar_chart(labels, values, title, ylabel):
    print(title)
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig

def create_gpu_memory_plots(fig_dir, config):
    metrics = load_metric_files(config['model_metric_file_paths'])
    create_grouped_bar_chart(
            labels=['Resnet34', 'Segformer b0'],
            values_by_subgroup={
                'Batch size 1':[
                    get_peak_gpu_memory(metrics, 'foundation training', 'resnet34', 1),
                    get_peak_gpu_memory(metrics, 'foundation training', 'segformer_b0', 1)
                ],
                'Batch size 2':[
                    get_peak_gpu_memory(metrics, 'foundation training', 'resnet34', 2),
                    get_peak_gpu_memory(metrics, 'foundation training', 'segformer_b0', 2)
                ]
            },
            title='Foundation Network Training Peak Memory Usage by Model',
            ylabel='Peak GPU Memory (GiB)').savefig(os.path.join(fig_dir, 'gpu_memory_foundation_training.png'))
    create_grouped_bar_chart(
            labels=['Resnet34 Siamese', 'Segformer b0 Siamese'],
            values_by_subgroup={
                'Batch size 1':[
                    get_peak_gpu_memory(metrics, 'flood training', 'resnet34_siamese', 1),
                    get_peak_gpu_memory(metrics, 'flood training', 'segformer_b0_siamese', 1)
                ],
                'Batch size 2':[
                    get_peak_gpu_memory(metrics, 'flood training', 'resnet34_siamese', 2),
                    get_peak_gpu_memory(metrics, 'flood training', 'segformer_b0_siamese', 2)
                ]
            },
            title='Flood Network Training Peak Memory Usage by Model',
            ylabel='Peak GPU Memory (GiB)').savefig(os.path.join(fig_dir, 'gpu_memory_flood_training.png'))
    create_bar_chart(
            labels=['Resnet34', 'Segformer b0'],
            values=[
                    get_peak_gpu_memory(metrics, 'foundation eval', 'resnet34'),
                    get_peak_gpu_memory(metrics, 'foundation eval', 'segformer_b0'),
                ],
            title='Foundation Inference Peak Memory Usage by Model',
            ylabel='Peak GPU Memory (GiB)').savefig(os.path.join(fig_dir, 'gpu_memory_foundation_eval.png'))
    create_bar_chart(
            labels=['Resnet34 Siamese', 'Segformer b0 Siamese'],
            values=[
                    get_peak_gpu_memory(metrics, 'flood eval', 'resnet34_siamese'),
                    get_peak_gpu_memory(metrics, 'flood eval', 'segformer_b0_siamese'),
                ],
            title='Flood Inference Peak Memory Usage by Model',
            ylabel='Peak GPU Memory (GiB)').savefig(os.path.join(fig_dir, 'gpu_memory_flood_eval.png'))


def run(config, fig_dir):
    peak_gpu_memory_plot_config = config.get('peak_gpu_memory_plot')
    if peak_gpu_memory_plot_config:
        create_gpu_memory_plots(fig_dir, peak_gpu_memory_plot_config)

if __name__ == '__main__':
    args = parse_args()
    run(load_json(args.config), args.fig_dir)
