import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

BASELINE = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASELINE)
from PIL import Image
from pathlib import Path
from datetime import datetime
import numpy as np

from utils.log import load_from_json

run_root = Path('/tmp/share/runs/spacenet8/nenad')

def plot_mIoU_vs_models():
    run_paths = ['/tmp/share/runs/spacenet8/nenad/2023-06-06-07-52_resnet34',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-08-06-25_resnet50',
                 '/tmp/share/runs/spacenet8/naijing/2023-06-07-01-29/dense_121', 
                 '/tmp/share/runs/spacenet8/naijing/2023-06-07-04-22/dense_161',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-06-06-05_segformer_b0',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-07-23-34_segformer_b1',
                 '/tmp/share/runs/spacenet8/nenad/2023-06-08-02-20_segformer_b2',
                 '/tmp/share/runs/spacenet8/naijing/2023-06-07-19-28/effunet_b2',
                 '/tmp/share/runs/spacenet8/naijing/2023-06-07-21-54/effunet_b4',
                 ]
    model_names = ['resnet34', 'resnet_50', 'dense-121', 'dense-161', 'segformer_b0', 'segformer_b1',  'segformer_b2', 'effunet_b2', 'effunet_b4']
    sample_foundation_images = {}
    sample_flood_images = {}
    result = pd.DataFrame()
    for run_path, model_name in zip(run_paths, model_names):
        metrics = load_from_json(os.path.join(run_path, 'metrics.json'))
        
        metrics_foundation = metrics['foundation training']
        metrics_flood = metrics['flood training']
        df1 = pd.DataFrame(metrics['foundation eval']['metrics_by_class']).transpose()
        df2 = pd.DataFrame(metrics['flood eval']['metrics_by_class']).transpose()
        df = pd.concat([df1, df2])
        df['model_name'] = model_name
        df['n_params'] = metrics_foundation['parameter_count'] + metrics_flood['parameter_count']
        df = df.rename_axis('class')
        df.reset_index(inplace=True)
        result = pd.concat([result, df])
        sample_foundation_images[model_name] = os.path.join(run_path, 'foundation/pngs/10300100AF395C00_2_16_51_pred.png')
        sample_flood_images[model_name] = os.path.join(run_path, 'flood/pngs/10300100AF395C00_2_16_51_pred.png')

    
    result = result[~result['class'].str.contains('road')]
    result['n_params'] = result['n_params'] * 1e-6
    result['iou'] = result['iou'] * 1e2
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)
    
    # # this is scatter plot but doesn't look well 
    # sns.set_theme(style="whitegrid")
    # ax = sns.scatterplot(data=result, x="n_params", y="iou", hue="index", style="model_name", s=100)
    
    result_agg = result.groupby('model_name')[['iou', 'n_params']].mean()  # .plot(x='n_params', y='iou', ax=ax, legend=True)
    
    # result['n_params'] = result['n_params_foundation'] + result['n_params_flood']
    # result['mean_iou'] = result[['building_iou', 'non_flooded_building_iou', 'flooded_building_iou']].mean(axis=1)
    
    for index, row in result_agg.iterrows():
        plt.scatter(row.n_params, row.iou, label=row.name)
        plt.annotate(row.name, (row.n_params+1, row.iou+0.5), fontsize=9)
        # plt.annotate(row.name, (row.n_params+1, row.iou+0.5), xytext=(row.n_params-10, row.iou+3), fontsize=9,
        #              arrowprops=dict(arrowstyle='->'))
    
    # ax.legend(bbox_to_anchor=(1.05, 1), ncol=1)
    ax[0].set_xlabel('Number of parameters (M)')
    ax[0].set_ylabel('mIoU (%)')
    ax[0].set_title('Mean IoU (buildings only)')
    ax[0].set_ylim(35, 60)
    ax[0].set_xlim(0, 170)
    # ax[0].legend(loc='upper right')
        
    # fig, ax = plt.subplots(figsize=(5, 3))
    
    df = result.pivot(index='model_name', columns='class', values='iou')
    df.reset_index(inplace=True)
    
    plt.subplot(2, 1, 2)
    df.plot(x='model_name', 
            y=['building', 'non-flooded building', 'flooded building'], 
            kind='bar', 
            ax=ax[1])
    plt.xticks(rotation=0)
    plt.xlabel('Encoder')
    plt.ylabel('IoU (%)')
    plt.title('IoU per class (buildings only)')
    plt.ylim(0,100)
    plt.legend(loc='upper right', ncol=2)
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    
    # plt.subplot(3, 1, 3)
    # df.plot(x='model_name', 
    #         y=['road', 'non-flooded road', 'flooded road'], 
    #         kind='bar', 
    #         ax=ax[1])
    # plt.xticks(rotation=0)
    # plt.xlabel('Encoder')
    # plt.ylabel('IoU (%)')
    # plt.title('IoU per class')
    # plt.ylim(0,100)
    # plt.legend(loc='upper right', ncol=2)
    # plt.xticks(rotation=15)
    # plt.grid(axis='y')
    
    
    plt.savefig(os.path.join(BASELINE, f'results/iou_vs_model.png'),
                dpi=300)
                # bbox_inches=Bbox.from_extents(-0.2, -0.2, 8, 4)
    
    plt.show()
    
    # for model_name in result.iterrows():
    # plt.scatter(x=result['n_params'], y=result['building_iou'], label='building')
    # result.scatter(x='n_params', y='non_flooded_building_iou', label='non-flooded building')
    
    # # plt.scatter(result['n_params'], result['building_iou'], marker='x', label='building')
    # # plt.scatter(row.n_params, row.non_flooded_building_iou, marker='o', label='non-flooded building')
    # # plt.scatter(row.n_params, row.flooded_building_iou, marker='.', label='flooded building')
    # # sns.scatterplot(data=result, x="total_bill", y="tip", hue="day", style="time")

    # plt.title('Mean IoU vs number of parameters\nSpaceNet8 validation dataset')
    # plt.xlabel('Number of parameters')
    # plt.ylabel('Mean IoU')
    # plt.grid()
    # plt.legend()
    # plot_name = f'result_{datetime.now().strftime("%d-%m-%Y-%H-%M")}.png'
    # plt.show()
    
    # df_result = result
    # df_result.plot(x='model_name_foundation', y=['building_iou', 'non_flooded_building_iou', 'flooded_building_iou'], kind='bar')
    # plt.xticks(rotation=0)
    # plt.ylabel('IoU (%)')
    # plt.legend(loc='lower right')
    # plt.gcf()
    # plt.show()
    
    return result, sample_foundation_images, sample_flood_images

def plot_sample_images(sample_images, fig_size=(10, 15), tag='foundation'):
    fig, axs = plt.subplots(len(sample_images), 1, figsize=fig_size)
    for i, (name, sample_image) in enumerate(sample_images.items()):
        image = Image.open(sample_image)
        print(name)
        plt.subplot(len(sample_images), 1, i+1)    
        plt.gcf()
        axs[i].imshow(image)
        axs[i].set_title(name, y=0.90)
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(BASELINE, f'results/sample_images_{tag}.png'),
                dpi=300)
                # bbox_inches=Bbox.from_extents(-0.2, -0.2, 8, 4)
    
    plt.show()
    
def crop_image(image_path, coords):
    image = Image.open(image_path)
    d = 290
    cropped_image = image.crop(coords + (coords[0] + d, coords[1] + d))
    return cropped_image
    