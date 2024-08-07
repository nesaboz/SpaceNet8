import csv
import os
import argparse
from datetime import datetime
import time

import torch
import torch.cuda
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models import foundation_models
pad_models = ['effunet_b2', 'effunet_b4', 'dense_121', 'dense_161']

from datasets.datasets import SN8Dataset
from core.losses import focal, soft_dice_loss
import models.pytorch_zoo.unet as unet
import models.other.segformer as segformer
import models.other.eff_pretrained as Densen_EffUnet

from models.other.unet import UNet
from utils.log import debug_msg, log_var_details, dump_command_line_args, TrainingMetrics

import inspect
from utils.utils import count_parameters
from utils.log import get_fcn_params, dump_to_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",
                        type=str,
                        required=True)
    parser.add_argument("--val_csv",
                         type=str,
                         required=True)
    parser.add_argument("--save_dir",
                         type=str,
                         required=True)
    parser.add_argument("--model_name",
                         type=str,
                         required=True)
    parser.add_argument("--lr",
                         type=float,
                        default=0.0001)
    parser.add_argument("--batch_size",
                         type=int,
                        default=2)
    parser.add_argument("--n_epochs",
                         type=int,
                         default=50)
    parser.add_argument("--gpu",
                        type=int,
                        default=0)
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None)
    return parser.parse_args()

# TODO: remove once TrainingMetrics persists each update
def write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv):
    epoch_dict = {"epoch":epoch}
    merged_metrics = {**epoch_dict, **train_metrics, **val_metrics}
    with open(training_log_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(merged_metrics)
        
def save_model_checkpoint(model, checkpoint_model_path): 
    torch.save(model.state_dict(), checkpoint_model_path)

models = foundation_models


def train_foundation(train_csv, val_csv, save_dir, model_name, initial_lr, batch_size, n_epochs, gpu, checkpoint_path=None, model_args={}, **kwargs):
    '''
    train_csv (str) - CSV files listing training examples
    val_csv (str) - CSV files listing validation examples
    save_dir (str) - directory to save model checkpoints, training logs, and other files.
    model_name (str) - type of model architecture to use
    initial_lr (float)- initial learning rate
    batch_size (int) - batch size
    n_epochs (int) - number of epochs to train for
    gpu (int) - which GPU to use
    checkpoint_path (str) - existing model weights to start training from
    model_args (dict) - Extra arguments to pass to the model constructor.
    **kwargs (dict) - extra optimizer arguments
    '''
    params = get_fcn_params(inspect.currentframe())
    
    img_size = (1300,1300)
    
    soft_dice_loss_weight = 0.25 # road loss
    focal_loss_weight = 0.75 # road loss
    road_loss_weight = 0.5
    building_loss_weight = 0.5

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    training_metrics = TrainingMetrics(model_name=model_name, batch_size=batch_size)
    training_metrics.start()

    SEED=12
    torch.manual_seed(SEED)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_model_path = os.path.join(save_dir, "model_checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    training_log_csv = os.path.join(save_dir, "log.csv")

    # init the training log
    with open(training_log_csv, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'lr', 'train_tot_loss', 'train_bce', 'train_dice', 'train_focal', 'train_road_loss',
                                     'val_tot_loss', 'val_bce', 'val_dice', 'val_focal', 'val_road_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    train_dataset = SN8Dataset(train_csv,
                            data_to_load=["preimg","building","roadspeed"],
                            img_size=img_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=batch_size)
    val_dataset = SN8Dataset(val_csv,
                            data_to_load=["preimg","building","roadspeed"],
                            img_size=img_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=4, batch_size=batch_size)
    training_metrics.record_dataset_metrics(train_dataset, val_dataset)

    #model = models["resnet34"](num_classes=[1, 8], num_channels=3)
    if model_name == "unet":
        model = UNet(3, [1,8], bilinear=True, **model_args)
    else:
        model = models[model_name](num_classes=[1, 8], num_channels=3, **model_args)  # there is 1 class for the building and 8 classes for the road, hence [1, 8]
    assert(hasattr(model, 'from_pretrained'))
    training_metrics.record_model_metrics(model)
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            step_size=kwargs.get('step_size', 40), gamma=kwargs.get('gamma', 0.5))
    bceloss = nn.BCEWithLogitsLoss()

    best_loss = np.inf
    
    if checkpoint_path:
        # load checkpoint
        print('Loaded checkpoint.')
        model_state_dict = torch.load(checkpoint_path)
        model.load_state_dict(model_state_dict)
    else:
        print('No checkpoint provided. Starting new training ...')

    # TODO: Delete once all scripts and notebooks are migrated from params.json
    # to metrics.json
    parameter_count = count_parameters(model)
    params.update({'parameter_count': parameter_count})
    dump_to_json(save_dir, params) 
    
    for epoch in range(n_epochs):
        print(f"EPOCH {epoch}")
        tic = time.time()

        ### Training ##
        model.train()
        train_loss_val = 0
        train_focal_loss = 0
        train_soft_dice_loss = 0
        train_bce_loss = 0
        train_road_loss = 0
        train_building_loss = 0
        epoch_duration_shorter = 0
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            preimg, postimg, building, road, roadspeed, flood = data

            preimg = preimg.cuda().float()
            roadspeed = roadspeed.cuda().float()
            building = building.cuda().float()
            tic1 = time.time()

            if model_name in pad_models:
                padded_preimg = torch.nn.functional.pad(preimg, (6, 6, 6, 6))
                padded_building_pred, padded_road_pred = model(padded_preimg)
                building_pred = padded_building_pred[..., 6:-6, 6:-6]
                road_pred = padded_road_pred[..., 6:-6, 6:-6]
            else:
                building_pred, road_pred = model(preimg)

            bce_l = bceloss(building_pred, building)
            y_pred = F.sigmoid(road_pred)

            focal_l = focal(y_pred, roadspeed)
            dice_soft_l = soft_dice_loss(y_pred, roadspeed)

            road_loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
            building_loss = bce_l
            loss = road_loss_weight * road_loss + building_loss_weight * building_loss

            train_loss_val+=loss
            train_focal_loss += focal_l
            train_soft_dice_loss += dice_soft_l
            train_bce_loss += bce_l
            train_road_loss += road_loss
            loss.backward()
            optimizer.step()
            
            toc1 = time.time()
            epoch_duration_shorter += toc1 - tic1
        
            if i == 0:
                debug_msg('Training loop vars')
                log_var_details('preimg', preimg)
                log_var_details('postimg', postimg)
                log_var_details('building', building)
                log_var_details('road', road)
                log_var_details('roadspeed', roadspeed)
                log_var_details('flood', flood)
                log_var_details('building_pred', building_pred)
                log_var_details('bce_l', bce_l)
                log_var_details('y_pred', y_pred)
                # preimg, Type: <class 'torch.Tensor'>, Shape: torch.Size([4, 3, 1300, 1300]), Dtype: torch.float32
                # postimg, Type: <class 'torch.Tensor'>, Shape: torch.Size([4]), Dtype: torch.int64
                # building, Type: <class 'torch.Tensor'>, Shape: torch.Size([4, 1, 1300, 1300]), Dtype: torch.float32
                # road, Type: <class 'torch.Tensor'>, Shape: torch.Size([4]), Dtype: torch.int64
                # roadspeed, Type: <class 'torch.Tensor'>, Shape: torch.Size([4, 8, 1300, 1300]), Dtype: torch.float32
                # flood, Type: <class 'torch.Tensor'>, Shape: torch.Size([4]), Dtype: torch.int64
                # building_pred, Type: <class 'torch.Tensor'>, Shape: torch.Size([4, 1, 1300, 1300]), Dtype: torch.float32
                # bce_l, Type: <class 'torch.Tensor'>, Shape: torch.Size([]), Dtype: torch.float32
                # y_pred, Type: <class 'torch.Tensor'>, Shape: torch.Size([4, 8, 1300, 1300]), Dtype: torch.float32


            print(f"    {str(np.round(i/len(train_dataloader)*100,2))}%: TRAIN LOSS: {(train_loss_val*1.0/(i+1)).item()}", end="\r")
        print()
        train_tot_loss = (train_loss_val*1.0/len(train_dataloader)).item()
        train_tot_focal = (train_focal_loss*1.0/len(train_dataloader)).item()
        train_tot_dice = (train_soft_dice_loss*1.0/len(train_dataloader)).item()
        train_tot_bce = (train_bce_loss*1.0/len(train_dataloader)).item()
        train_tot_road_loss = (train_road_loss*1.0/len(train_dataloader)).item()
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        train_metrics = {"lr":current_lr, "train_tot_loss":train_tot_loss,
                         "train_bce":train_tot_bce, "train_focal":train_tot_focal,
                         "train_dice":train_tot_dice, "train_road_loss":train_tot_road_loss}

        # validation
        model.eval()
        val_loss_val = 0
        val_focal_loss = 0
        val_soft_dice_loss = 0
        val_bce_loss = 0
        val_road_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                preimg, postimg, building, road, roadspeed, flood = data
                preimg = preimg.cuda().float()
                roadspeed = roadspeed.cuda().float()
                building = building.cuda().float()

                if model_name in pad_models:
                    padded_preimg = torch.nn.functional.pad(preimg, (6, 6, 6, 6))
                    padded_building_pred, padded_road_pred = model(padded_preimg)
                    building_pred = padded_building_pred[..., 6:-6, 6:-6]
                    road_pred = padded_road_pred[..., 6:-6, 6:-6]

                else:
                    building_pred, road_pred = model(preimg)

                bce_l = bceloss(building_pred, building)
                y_pred = F.sigmoid(road_pred)

                focal_l = focal(y_pred, roadspeed)
                dice_soft_l = soft_dice_loss(y_pred, roadspeed)

                road_loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
                building_loss = bce_l
                loss = road_loss_weight * road_loss + building_loss_weight * building_loss

                val_focal_loss += focal_l
                val_soft_dice_loss += dice_soft_l
                val_bce_loss += bce_l
                val_loss_val += loss
                val_road_loss += road_loss

                print(f"    {str(np.round(i/len(val_dataloader)*100,2))}%: VAL LOSS: {(val_loss_val*1.0/(i+1)).item()}", end="\r")

        print()        
        val_tot_loss = (val_loss_val*1.0/len(val_dataloader)).item()
        val_tot_focal = (val_focal_loss*1.0/len(val_dataloader)).item()
        val_tot_dice = (val_soft_dice_loss*1.0/len(val_dataloader)).item()
        val_tot_bce = (val_bce_loss*1.0/len(val_dataloader)).item()
        val_tot_road_loss = (val_road_loss*1.0/len(val_dataloader)).item()
        val_metrics = {"val_tot_loss":val_tot_loss,"val_bce":val_tot_bce,
                       "val_focal":val_tot_focal, "val_dice":val_tot_dice, "val_road_loss":val_tot_road_loss}

        write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv)

        save_model_checkpoint(model, checkpoint_model_path)

        toc = time.time()
        epoch_duration = toc - tic
        print(f"Epoch took: {epoch_duration/60.0:.1f} minutes")
        print(f"Forward/backward pass took: {epoch_duration_shorter} seconds")
        
        training_metrics.add_epoch({**train_metrics, **val_metrics, 
                                    'epoch_duration': epoch_duration, 
                                    'epoch_duration_shorter': epoch_duration_shorter})

        epoch_val_loss = val_metrics["val_tot_loss"]
        if epoch_val_loss < best_loss:
            print(f"    loss improved from {np.round(best_loss, 6)} to {np.round(epoch_val_loss, 6)}. saving best model...")
            best_loss = epoch_val_loss
            save_model_checkpoint(model, best_model_path)
    training_metrics.end()
    return training_metrics

if __name__ == "__main__":
    args = parse_args()
    train_csv = args.train_csv
    val_csv = args.val_csv
    save_dir = args.save_dir
    model_name = args.model_name
    initial_lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    gpu = args.gpu
    checkpoint_path = args.checkpoint

    # dump_command_line_args(os.path.join(save_dir, 'args.txt'))
    train_foundation(train_csv, val_csv, save_dir, model_name, initial_lr, batch_size, n_epochs, gpu, checkpoint_path)
