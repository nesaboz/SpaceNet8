import os
import argparse
import time
import datetime

from osgeo import gdal
from osgeo import osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn

from models import foundation_models
from train_foundation_features import models

import models.pytorch_zoo.unet as unet
from models.other.unet import UNet
import models.other.segformer as segformer
from datasets.datasets import SN8Dataset
from utils.log import get_eval_results_path
from utils.log import write_to_csv_file
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
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--save_preds_dir",
                         type=str,
                         required=False,
                         default=None)
    parser.add_argument("--gpu",
                         type=int,
                         required=False,
                         default=0)
    return parser.parse_args()

def make_prediction_png_roads_buildings(image, gts, predictions, save_figure_filename):
    bldg_gt = gts[0][0]
    road_gt = gts[1]
    bldg_pred = predictions[0][0]
    road_pred = predictions[1]
    # print("bldg gt shape: ", bldg_gt.shape)
    # print("road gt shape: ", road_gt.shape)
    # print("bldg pred shape: ", bldg_pred.shape)
    # print("road pred shape: ", road_pred.shape)
    
    # seperate the binary road preds and speed preds
    binary_road_pred = road_pred[-1]
    binary_road_gt = road_gt[-1]
    
    speed_pred = np.argmax(road_pred[:-1], axis=0)
    speed_gt = np.argmax(road_gt[:-1], axis=0) 
    
    roadspeed_shape = road_pred.shape
    tempspeed = np.zeros(shape=(roadspeed_shape[0]+1,roadspeed_shape[1],roadspeed_shape[2]))
    tempspeed[1:] = road_pred
    road_pred = tempspeed
    road_pred = np.argmax(road_pred, axis=0)
    
    combined_pred = np.zeros(shape=bldg_pred.shape, dtype=np.uint8)
    combined_pred = np.where(bldg_pred==1, 1, combined_pred)
    combined_pred = np.where(binary_road_pred==1, 2, combined_pred)
    
    combined_gt = np.zeros(shape=bldg_gt.shape, dtype=np.uint8)
    combined_gt = np.where(bldg_gt==1, 1, combined_gt)
    combined_gt = np.where(binary_road_gt==1, 2, combined_gt)
    
    
    raw_im = np.moveaxis(image, 0, -1) # now it is channels last
    raw_im = raw_im/np.max(raw_im)
    
    grid = [[raw_im, combined_gt, combined_pred, speed_gt, speed_pred]]
    
    nrows = len(grid)
    ncols = len(grid[0])
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*4,nrows*4))
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[col]
            ax.axis('off')
            if row==0 and col==0:
                ax.imshow(grid[row][col])
            elif row==0 and col in [3,4]:
                combined_mask_cmap = colors.ListedColormap(['black', 'green', 'blue', 'red',
                                                            'purple', 'orange', 'yellow', 'brown',
                                                            'pink'])
                ax.imshow(grid[row][col], cmap=combined_mask_cmap, interpolation='nearest', origin='upper',
                                  norm=colors.BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7, 8], combined_mask_cmap.N))
            if row==0 and col in [1,2]:
                combined_mask_cmap = colors.ListedColormap(['black', 'red', 'blue'])
                ax.imshow(grid[row][col],
                          interpolation='nearest', origin='upper',
                          cmap=combined_mask_cmap,
                          norm=colors.BoundaryNorm([0, 1, 2, 3], combined_mask_cmap.N))
            # if row==1 and col == 1:
            #     ax.imshow(grid[0][0])
            #     mask = np.where(combined_gt==0, np.nan, combined_gt)
            #     ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
            # if row==1 and col == 2:
            #     ax.imshow(grid[0][0])
            #     mask = np.where(combined_pred==0, np.nan, combined_pred)
            #     ax.imshow(mask, cmap='gist_rainbow_r', alpha=0.6)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_figure_filename)
    plt.close(fig)
    plt.close('all')

models = foundation_models

def foundation_eval(model_path, in_csv, save_fig_dir, save_preds_dir, model_name, gpu=0, create_folders=True):
    """
    We run evaluation on validation data to generate tiff images (segmentation masks) and pngs (for visualization). Note: these are geotiff images (not tiff), and to load them one must use osgeo.gdal (see `SN8Dataset.__getitem__`). 

    - building segmentation mask (geotiff) has 1 or 0 for building
    """
    if create_folders and save_fig_dir:
        os.makedirs(save_fig_dir)
    if create_folders and save_preds_dir:
        os.makedirs(save_preds_dir)
    
    img_size = (1300,1300)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    eval_metrics = EvalMetrics(model_name=model_name)
    eval_metrics.start()

    if model_name == "unet":
        model = UNet(3, [1,8], bilinear=True)
    else:
        model = models[model_name](num_classes=[1, 8], num_channels=3)
    model.eval()
    eval_metrics.record_model_metrics(model)
    val_dataset = SN8Dataset(in_csv,
                        data_to_load=["preimg","building","roadspeed"],
                        img_size=img_size)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)  # note that the batch size is 1 here, it simplifies evaluation
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    #criterion = nn.BCEWithLogitsLoss()

    running_tp = [0,0] 
    running_fp = [0,0]
    running_fn = [0,0]
    running_union = [0,0]

    filenames = [[], []]
    precisions = [[], []]
    recalls = [[], []]
    f1s = [[], []]
    ious = [[], []]
    positives = [[], []]

    val_loss_val = 0

    eval_results_file = get_eval_results_path(save_fig_dir, save_preds_dir)
            
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            predictions = np.zeros((2,8,img_size[0],img_size[1]))
            gts = np.zeros((2,8,img_size[0],img_size[1]))
            current_image_filename = val_dataset.get_image_filename(i)
            print("evaluating: ", i, os.path.basename(current_image_filename))
            preimg, postimg, building, road, roadspeed, flood = data
            preimg = preimg.cuda().float()
            
            roadspeed = roadspeed.cuda().float()
            building = building.cuda().float()


            # building_pred, roadspeed_pred = model(preimg)
            pad_models = ['effunet_b2', 'effunet_b4', 'dense_121', 'dense_161']

            if model_name in pad_models:
                padded_preimg = torch.nn.functional.pad(preimg, (6, 6, 6, 6))
                padded_building_pred, padded_road_pred = model(padded_preimg)
                building_pred = padded_building_pred[..., 6:-6, 6:-6]
                roadspeed_pred = padded_road_pred[..., 6:-6, 6:-6]

            else:
                building_pred, roadspeed_pred = model(preimg)

            roadspeed_pred = torch.sigmoid(roadspeed_pred)
            building_pred = torch.sigmoid(building_pred)
            
            preimg = preimg.cpu().numpy()[0] # index at 0 so we have (C,H,W)
            gt_building = building.cpu().numpy()[0][0] # index so building gt is (H, W)
            gt_roadspeed = roadspeed.cpu().numpy()[0] # index so we have (C,H,W)
            
            
            building_prediction = building_pred.cpu().numpy()[0][0] # index so shape is (H,W) for buildings
            building_prediction = np.rint(building_prediction).astype(int)
            road_prediction = roadspeed_pred.cpu().numpy()[0] # index so we have (C,H,W)
            roadspeed_prediction = np.rint(road_prediction).astype(int)
            
            gts[0,0] = gt_building
            gts[1,:] = gt_roadspeed
            predictions[0,0] = building_prediction
            predictions[1,:] = roadspeed_prediction

            ### save prediction
            if save_preds_dir is not None:
                road_pred_arr = (road_prediction * 255).astype(np.uint8) # to be compatible with the SN5 eval and road speed prediction, need to mult by 255
                ds = gdal.Open(current_image_filename)
                geotran = ds.GetGeoTransform()
                xmin, xres, rowrot, ymax, colrot, yres = geotran
                raster_srs = osr.SpatialReference()
                raster_srs.ImportFromWkt(ds.GetProjectionRef())
                ds = None
                output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_roadspeedpred.tif")))
                nchannels, nrows, ncols = road_pred_arr.shape
                write_geotiff(output_tif, ncols, nrows,
                            xmin, xres, ymax, yres,
                            raster_srs, road_pred_arr)
                            
                building_pred_arr = np.array([(building_prediction * 255).astype(np.uint8)])
                output_tif = os.path.join(save_preds_dir, os.path.basename(current_image_filename.replace(".tif","_buildingpred.tif")))
                nchannels, nrows, ncols = road_pred_arr.shape
                write_geotiff(output_tif, ncols, nrows,
                            xmin, xres, ymax, yres,
                            raster_srs, building_pred_arr)
            
            for j in range(len(gts)): # iterate through the building and road gt, i.e. for j in [0, 1]
                prediction = predictions[j]
                gt = gts[j]
                if j == 1: # it's roadspeed, so get binary pred and gt for metrics
                    prediction = prediction[-1]
                    gt = gt[-1]
                
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
            
                current_image_filename = val_dataset.files[i]["preimg"]
                filenames[j].append(current_image_filename)
                if np.sum(gt) < 1:
                    positives[j].append("n")
                else:
                    positives[j].append("y") 

            if save_fig_dir is not None:
                #if save_preds_dir is not None: # for some reason, seg fault when doing both of these. maybe file saving or something is interfering. so sleep for a little
                #    time.sleep(2) 
                save_figure_filename = os.path.join(save_fig_dir, os.path.basename(current_image_filename)[:-4]+"_pred.png")
                make_prediction_png_roads_buildings(preimg, gts, predictions, save_figure_filename)
    
    print()
    data = ["building", "road"]
    datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for i in range(len(running_tp)):
        print(f"final metrics for: {data[i]}")
        precision = running_tp[i] / (running_tp[i] + running_fp[i] + 0.00001)
        recall = running_tp[i] / (running_tp[i] + running_fn[i] + 0.00001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
        iou = running_tp[i] / (running_union[i] + 0.00001)
        print("final running evaluation score: ")
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1: ", f1)
        print("iou: ", iou)
        print()

        eval_metrics.add_class_metrics(data[i],
                {'precision':precision, 'recall':recall, 'f1':f1, 'iou':iou})
        write_to_csv_file(datetime_str, model_name, data[i], precision, recall, f1, iou, eval_results_file)
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

    foundation_eval(model_path, in_csv, save_fig_dir, save_preds_dir, model_name, gpu)
