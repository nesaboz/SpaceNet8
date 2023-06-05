import os
import time
import argparse
import sys

from osgeo import gdal
from osgeo import osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.cuda
import torch.nn as nn
import psutil
import datetime

from foundation_eval import write_to_csv_file, get_eval_results_path

import models.pytorch_zoo.unet as unet
from datasets.datasets import SN8Dataset
from models.other.unet import UNetSiamese
import models.other.segformer as segformer
from utils.log import EvalMetrics
from utils.utils import write_geotiff

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                         type=str,
                         required=True)
    parser.add_argument("--model_name",
                         type=str,
                         required=True)
    parser.add_argument("--in_csv",
                       type=str,
                       required=True)
    parser.add_argument("--save_fig_dir",
                        help="saves model predictions as .pngs for easy viewing.",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--save_preds_dir",
                        help="saves model predictions as .tifs",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--gpu",
                         type=int,
                         required=False,
                         default=0)
    return parser.parse_args()


def print_top_memory_variables(local_vars, var_number_to_print=5):
    """Prints top variables in terms of memory. 
    Usage: `print_top_memory_variables(locals().copy())` can't call locals() in the function itself.

    Args:
        local_vars (dict): pass `locals().copy()` 
        var_number_to_print(int): 
    """

    # Get the local variables
    memory = {}

    # Iterate over the local variables and print their sizes
    for var_name, var_value in local_vars.items():
        var_size = sys.getsizeof(var_value)
        memory[var_name] = var_size
        
    memory_sorted = sorted(memory.items(), key=lambda x: x[1], reverse=True)[:var_number_to_print]
    
    for (var_name, var_size) in memory_sorted:
        print(f"Variable: {var_name}, Size: {var_size} bytes")
        
        

def print_cpu_memory(verbose=False):

    # Get the current memory usage
    memory_info = psutil.virtual_memory()

    # Extract the memory information
    total_memory = memory_info.total
    available_memory = memory_info.available
    used_memory = memory_info.used
    percent_memory = memory_info.percent

    # Convert bytes to megabytes
    total_memory_mb = total_memory / 1024**2
    available_memory_mb = available_memory / 1024**2
    used_memory_mb = used_memory / 1024**2

    if verbose:
        # Print the memory information
        print(f"Total CPU memory: {total_memory_mb:.2f} MB")
        print(f"Available CPU memory: {available_memory_mb:.2f} MB")
        print(f"Used CPU memory: {used_memory_mb:.2f} MB")
        print(f"Percentage of used CPU memory: {percent_memory}%")
    else:
        print(f"Percentage of used CPU memory: {percent_memory}%")
        
        

def make_prediction_png(image, postimage, gt, prediction, save_figure_filename):
    #raw_im = image[:,:,:3]
    #raw_im = np.asarray(raw_im[:,:,::-1], dtype=np.float32)
    raw_im = np.moveaxis(image, 0, -1) # now it is channels last
    raw_im = raw_im/np.max(raw_im)
    post_im = np.moveaxis(postimage, 0, -1)
    post_im = post_im/np.max(post_im)
    
    #gt = np.asarray(gt*255., dtype=np.uint8)
    #pred = np.asarray(prediction*255., dtype=np.uint8)
    
    combined_mask_cmap = colors.ListedColormap(['black', 'red', 'blue', 'green', 'yellow'])

    grid = [[raw_im, gt, prediction],[post_im, 0, 0]]

    fig, axs = plt.subplots(2, 3, figsize=(12,8))
    for row in range(2):
        for col in range(3):
            ax = axs[row][col]
            ax.axis('off')
            if row==0 and col == 0:
                theim = ax.imshow(grid[row][col])
            elif row==1 and col == 0:
                theim = ax.imshow(grid[row][col])
            elif row==0 and col in [1,2]:
                ax.imshow(grid[row][col],
                          interpolation='nearest', origin='upper',
                          cmap=combined_mask_cmap,
                          norm=colors.BoundaryNorm([0, 1, 2, 3, 4, 5], combined_mask_cmap.N))
            elif row==1 and col == 1:
                ax.imshow(grid[0][0])
                mask = np.where(gt==0, np.nan, 1)
                ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
            elif row==1 and col == 2:
                ax.imshow(grid[0][0])
                mask = np.where(prediction==0, np.nan, 1)
                ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_figure_filename, dpi=95)
    plt.close(fig)
    plt.close('all')
                
    
models = {
    'resnet34_siamese': unet.Resnet34_siamese_upsample,
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,
    'unet_siamese': UNetSiamese,
    'segformer_b0_siamese': segformer.SiameseSegformer_b0,
    'segformer_b1_siamese': segformer.SiameseSegformer_b1,
}

def flood_eval(model_path, in_csv, save_fig_dir, save_preds_dir, model_name, gpu=0, create_folders=True):
    """
    We run evaluation on validation data to generate tiff images (segmentation masks) and pngs (for visualization). Note: these are geotiff images (not tiff), and to load them one must use osgeo.gdal (see `SN8Dataset.__getitem__`). 

    - flood segmentation mask has: 0 for background, 1 non-flooded building, 2 is flooded building, 3 is non-flooded road, 4 is flooded road
    """
    if create_folders and save_fig_dir:
        os.makedirs(save_fig_dir)
    if create_folders and save_preds_dir:
        os.makedirs(save_preds_dir)
    
    num_classes = 5
    img_size = (1300,1300)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    eval_metrics = EvalMetrics(model_name=model_name)
    eval_metrics.start()

    val_dataset = SN8Dataset(in_csv,
                             data_to_load=["preimg","postimg","flood"],
                             img_size=img_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    if model_name == "unet_siamese":
        model = UNetSiamese(3, num_classes, bilinear=True)
    else:
        model = models[model_name](num_classes=num_classes, num_channels=3)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    eval_metrics.record_model_metrics(model)

    #criterion = nn.BCEWithLogitsLoss()

    # predictions = np.zeros((len(val_dataset),img_size[0],img_size[1]))
    # gts = np.zeros((len(val_dataset),img_size[0],img_size[1]))

    # we need running numbers for each class: [no flood bldg, flood bldg, no flood road, flood road]
    classes = ["non-flooded building", "flooded building", "non-flooded road", "flooded road"]
    running_tp = [0, 0, 0, 0]
    running_fp = [0, 0, 0, 0]
    running_fn = [0, 0, 0, 0]
    running_union = [0, 0, 0, 0]

    filenames = []
    precisions = [[],[],[],[]]
    recalls = [[],[],[],[]]
    f1s = [[],[],[],[]]
    ious = [[],[],[],[]]
    positives = [[],[],[],[]]
    
    eval_results_file = get_eval_results_path(save_fig_dir, save_preds_dir)
    
    val_loss_val = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            
            # print_top_memory_variables(locals().copy())
            
            current_image_filename = val_dataset.get_image_filename(i)
            print("evaluating: ", i, os.path.basename(current_image_filename))
            preimg, postimg, building, road, roadspeed, flood = data

            # print_cpu_memory()
            
            preimg = preimg.cuda().float() #siamese
            postimg = postimg.cuda().float() #siamese
            
            flood = flood.numpy()
            flood_shape = flood.shape
            flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
            flood = np.argmax(flood, axis = 1)
            
            flood = torch.tensor(flood).cuda()

            flood_pred = model(preimg, postimg) # siamese resnet34 with stacked preimg+postimg input
            flood_pred = torch.nn.functional.softmax(flood_pred, dim=1).cpu().numpy()[0] # (5, H, W)
            #for i in flood_pred:
            #    plt.imshow(i)
            #    plt.colorbar()
            #    plt.show()
            
            flood_prediction = np.argmax(flood_pred, axis=0) # (H, W)
            #plt.imshow(flood_pred)
            #plt.colorbar()
            #plt.show()
            
            #flood_pred = torch.softmax(flood_pred)
            #flood_pred = torch.sigmoid(flood_pred)
            
            #print(flood_pred.shape)
            
            ### save prediction
            if save_preds_dir is not None:
                ds = gdal.Open(current_image_filename)
                geotran = ds.GetGeoTransform()
                xmin, xres, rowrot, ymax, colrot, yres = geotran
                raster_srs = osr.SpatialReference()
                raster_srs.ImportFromWkt(ds.GetProjectionRef())
                ds = None
                output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_floodpred.tif")))
                nrows, ncols = flood_prediction.shape
                write_geotiff(output_tif, ncols, nrows,
                            xmin, xres, ymax, yres,
                            raster_srs, [flood_prediction])
            
            preimg = preimg.cpu().numpy()[0] # index at 0 so we have (C,H,W)
            postimg = postimg.cpu().numpy()[0]
            
            gt_flood = flood.cpu().numpy()[0] # index so building gt is (H, W)
            
            #flood_prediction = flood_pred.cpu().numpy()[0] # index so shape is (C,H,W) for buildings
            #flood_prediction = np.append(np.zeros(shape=(1,flood_shape[2],flood_shape[3])), flood_prediction, axis=0) # for focal loss 
            #flood_prediction = np.argmax(flood_prediction, axis=0)
            #flood_prediction = np.rint(flood_prediction).astype(int)

            for j in range(4): # there are 4 classes
                gt = np.where(gt_flood==(j+1), 1, 0) # +1 because classes start at 1. background is 0
                prediction = np.where(flood_prediction==(j+1), 1, 0)
                
                #gts[i] = gt_flood
                #predictions[i] = flood_prediction

                tp = np.rint(prediction * gt)
                fp = np.rint(prediction - tp)
                fn = np.rint(gt - tp)
                union = np.rint(np.sum(prediction + gt - tp))

                iou = np.sum(tp) / np.sum((prediction + gt - tp + 0.00001))
                tp = np.sum(tp).astype(int)
                fp = np.sum(fp).astype(int)
                fn = np.sum(fn).astype(int)

                running_tp[j]+=tp
                running_fp[j]+=fp
                running_fn[j]+=fn
                running_union[j]+=union

                #acc = np.sum(np.where(prediction == gt, 1, 0)) / (gt.shape[0] * gt.shape[1])
                precision = tp / (tp + fp + 0.00001)
                recall = tp / (tp + fn + 0.00001)
                f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
                precisions[j].append(precision)
                recalls[j].append(recall)
                f1s[j].append(f1)
                ious[j].append(iou)

                if np.sum(gt) < 1:
                    positives[j].append("n")
                else:
                    positives[j].append("y")
                
            current_image_filename = val_dataset.files[i]["preimg"]
            filenames.append(current_image_filename)
            if save_fig_dir != None:
                save_figure_filename = os.path.join(save_fig_dir, os.path.basename(current_image_filename)[:-4]+"_pred.png")
                make_prediction_png(preimg, postimg, gt_flood, flood_prediction, save_figure_filename)

    print()
    
    datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for j in range(len(classes)):
        print(f"class: {classes[j]}")
        precision = running_tp[j] / (running_tp[j] + running_fp[j] + 0.00001)
        recall = running_tp[j] / (running_tp[j] + running_fn[j] + 0.00001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
        iou = running_tp[j] / (running_union[j] + 0.00001)
        print("  precision: ", precision)
        print("  recall: ", recall)
        print("  f1: ", f1)
        print("  iou: ", iou)
        eval_metrics.add_class_metrics(classes[j],
                {'precision':precision, 'recall':recall, 'f1':f1, 'iou':iou})
        
        write_to_csv_file(datetime_str, model_name, classes[j], precision, recall, f1, iou, eval_results_file)
    eval_metrics.end()
    return eval_metrics
        

if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    in_csv = args.in_csv
    save_fig_dir = args.save_fig_dir
    save_preds_dir = args.save_preds_dir
    model_name = args.model_name
    gpu = args.gpu
    
    flood_eval(model_path, in_csv, save_fig_dir, save_preds_dir, model_name, gpu)
