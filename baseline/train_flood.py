import csv
import os
import argparse
import datetime
from datetime import datetime
import time
import psutil

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

from datasets.datasets import SN8Dataset
import models.pytorch_zoo.unet as unet
from models.other.unet import UNetSiamese
import models.other.segformer as segformer
from models.other.siamunetdif import SiamUnet_diff
from models.other.siamnestedunet import SNUNet_ECAM
from utils.log import debug_msg, log_var_details, dump_command_line_args, TrainingMetrics
import models.other.eff_pretrained as Densen_EffUnet


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
    parser.add_argument("--from_pretrained",
                         action='store_true',
                         help='Initialize the model with pretrained weights')
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

# TODO: remove once flood training metrics persists each update
def write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv):
    epoch_dict = {"epoch":epoch}
    merged_metrics = {**epoch_dict, **train_metrics, **val_metrics}
    with open(training_log_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(merged_metrics)
        
def save_model_checkpoint(model, checkpoint_model_path): 
    torch.save(model.state_dict(), checkpoint_model_path)
        
def save_best_model(model, best_model_path):
    torch.save(model.state_dict(), best_model_path)

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
    # No pretrained weights available
    'unet_siamese':UNetSiamese,
    # No pretrained weights available
    'unet_siamese_dif':SiamUnet_diff,
    # No pretrained weights available
    'nestedunet_siamese':SNUNet_ECAM,
    'segformer_b0_siamese': segformer.SiameseSegformer_b0,
    'segformer_b1_siamese': segformer.SiameseSegformer_b1,
    'effunet_b2_siamese': Densen_EffUnet.EffUnet_b2,
    'effunet_b4_siamese': Densen_EffUnet.EffUnet_b4,
    'dense_121_siamese': Densen_EffUnet.Dense_121,
    'dense_161_siamese': Densen_EffUnet.Dense_161
}

def train_flood(train_csv, val_csv, save_dir, model_name, initial_lr, batch_size, n_epochs, gpu, checkpoint_path=None, model_args={}, **kwargs):
    '''
    train_csv - CSV files listing training examples
    val_csv - CSV files listing validation examples
    save_dir - directory to save model checkpoints, training logs, and other files.
    model_name - type of model architecture to use
    initial_lr - initial learning rate
    batch_size - batch size
    n_epochs - number of epochs to train for
    gpu - which GPU to use
    checkpoint_path - existing model weights to start training from
    model_args - Extra arguments to pass to the model constructor.
    **kwargs - extra arguments
    '''
    
    params = get_fcn_params(inspect.currentframe())

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    training_metrics = TrainingMetrics(model_name=model_name, batch_size=batch_size)
    training_metrics.start()

    soft_dice_loss_weight = 0.25
    focal_loss_weight = 0.75
    num_classes=5
    class_weights = None

    road_loss_weight = 0.5
    building_loss_weight = 0.5

    img_size = (1300,1300)

    SEED=12
    torch.manual_seed(SEED)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_model_path = os.path.join(save_dir, "model_checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    training_log_csv = os.path.join(save_dir, "log.csv")

    # init the training log
    with open(training_log_csv, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'lr', 'train_tot_loss',
                                     'val_tot_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    train_dataset = SN8Dataset(train_csv,
                            data_to_load=["preimg","postimg","flood"],
                            img_size=img_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=batch_size)
    val_dataset = SN8Dataset(val_csv,
                            data_to_load=["preimg","postimg","flood"],
                            img_size=img_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=2, batch_size=batch_size)
    training_metrics.record_dataset_metrics(train_dataset, val_dataset)

    #model = models["resnet34"](num_classes=5, num_channels=6)
    if model_name == "unet_siamese":
        # No pretrained weights available
        model = UNetSiamese(3, num_classes, bilinear=True, **model_args)
    else:
        model = models[model_name](num_classes=num_classes, num_channels=3, **model_args)  # num classes here is 5, (0: background, 1: non-floded building, 2: flooded building, 3: non-flooded road, and 4: flooded road)
    assert(hasattr(model, 'from_pretrained'))
    training_metrics.record_model_metrics(model)

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            step_size=kwargs.get('step_size', 40), gamma=kwargs.get('gamma', 0.5))
    
    if class_weights is None:
        celoss = nn.CrossEntropyLoss()
    else:
        celoss = nn.CrossEntropyLoss(weight=class_weights)

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
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
                
            preimg, postimg, building, road, roadspeed, flood = data

            preimg = preimg.cuda().float()
            postimg = postimg.cuda().float()

            flood = flood.numpy()
            flood_shape = flood.shape
            flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
            flood = np.argmax(flood, axis = 1) # this is needed for cross-entropy loss. 

            flood = torch.tensor(flood).cuda()

            pad_models = ['effunet_b2', 'effunet_b4', 'dense_121', 'dense_161']

            if model_name in pad_models:
                combinedimg = torch.cat((preimg, postimg), dim=1)
                combinedimg = combinedimg.cuda().float()
                padded_combinedimg = torch.nn.functional.pad(combinedimg, (6, 6, 6, 6))
                padded_flood_pred = model(padded_combinedimg) # stacked preimg+postimg input
                flood_pred = padded_flood_pred[..., 6:-6, 6:-6]
            
            else:
                flood_pred = model(preimg, postimg) # this is for siamese resnet34 with stacked preimg+postimg input

            #y_pred = F.sigmoid(flood_pred)
            #focal_l = focal(y_pred, flood)
            #dice_soft_l = soft_dice_loss(y_pred, flood)
            #loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)
            loss = celoss(flood_pred, flood.long())
            
            # if i % 10 == 0:
            #     print_gpu_memory()

            train_loss_val+=loss
            #train_focal_loss += focal_l
            #train_soft_dice_loss += dice_soft_l
            loss.backward()
            optimizer.step()
            if i == 0:
                debug_msg('Training loop vars')
                log_var_details('preimg', preimg)
                log_var_details('postimg', postimg)
                log_var_details('building', building)
                log_var_details('road', road)
                log_var_details('roadspeed', roadspeed)
                log_var_details('flood', flood)
                log_var_details('flood_pred', flood_pred)
                # preimg, Type: <class 'torch.Tensor'>, Shape: torch.Size([2, 3, 1300, 1300]), Dtype: torch.float32
                # postimg, Type: <class 'torch.Tensor'>, Shape: torch.Size([2, 3, 1300, 1300]), Dtype: torch.float32
                # building, Type: <class 'torch.Tensor'>, Shape: torch.Size([2]), Dtype: torch.int64
                # road, Type: <class 'torch.Tensor'>, Shape: torch.Size([2]), Dtype: torch.int64
                # roadspeed, Type: <class 'torch.Tensor'>, Shape: torch.Size([2]), Dtype: torch.int64
                # flood, Type: <class 'torch.Tensor'>, Shape: torch.Size([2, 1300, 1300]), Dtype: torch.int64
                # flood_pred, Type: <class 'torch.Tensor'>, Shape: torch.Size([2, 5, 1300, 1300]), Dtype: torch.float32

            print(f"    {str(np.round(i/len(train_dataloader)*100,2))}%: TRAIN LOSS: {(train_loss_val*1.0/(i+1)).item()}", end="\r")
        print()
        train_tot_loss = (train_loss_val*1.0/len(train_dataloader)).item()
        #train_tot_focal = (train_focal_loss*1.0/len(train_dataloader)).item()
        #train_tot_dice = (train_soft_dice_loss*1.0/len(train_dataloader)).item()
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        train_metrics = {"lr":current_lr, "train_tot_loss":train_tot_loss}

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

                combinedimg = torch.cat((preimg, postimg), dim=1)
                combinedimg = combinedimg.cuda().float()
                preimg = preimg.cuda().float()
                postimg = postimg.cuda().float()

                flood = flood.numpy()
                flood_shape = flood.shape
                flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
                flood = np.argmax(flood, axis = 1) # for crossentropy
                
                #temp = np.zeros(shape=(flood_shape[0],6,flood_shape[2],flood_shape[3]))
                #temp[:,:4] = flood
                #temp[:,4] = np.max(flood[:,:2], axis=1)
                #temp[:,5] = np.max(flood[:,2:], axis=1)
                #flood = temp
                

                flood = torch.tensor(flood).cuda()
                pad_models = ['effunet_b2', 'effunet_b4', 'dense_121', 'dense_161']

                if model_name in pad_models:
                    combinedimg = torch.cat((preimg, postimg), dim=1)
                    combinedimg = combinedimg.cuda().float()
                    padded_combinedimg = torch.nn.functional.pad(combinedimg, (6, 6, 6, 6))
                    padded_flood_pred = model(padded_combinedimg) # stacked preimg+postimg input
                    flood_pred = padded_flood_pred[..., 6:-6, 6:-6]
                
                else:
                    flood_pred = model(preimg, postimg) 


                #y_pred = F.sigmoid(flood_pred)
                #focal_l = focal(y_pred, flood)
                #dice_soft_l = soft_dice_loss(y_pred, flood)
                #loss = (focal_loss_weight * focal_l + soft_dice_loss_weight * dice_soft_l)

                loss = celoss(flood_pred, flood.long())

                #val_focal_loss += focal_l
                #val_soft_dice_loss += dice_soft_l
                val_loss_val += loss

                print(f"    {str(np.round(i/len(val_dataloader)*100,2))}%: VAL LOSS: {(val_loss_val*1.0/(i+1)).item()}", end="\r")

        print()        
        val_tot_loss = (val_loss_val*1.0/len(val_dataloader)).item()
        #val_tot_focal = (val_focal_loss*1.0/len(val_dataloader)).item()
        #val_tot_dice = (val_soft_dice_loss*1.0/len(val_dataloader)).item()
        val_metrics = {"val_tot_loss":val_tot_loss}

        write_metrics_epoch(epoch, fieldnames, train_metrics, val_metrics, training_log_csv)

        save_model_checkpoint(model, checkpoint_model_path)

        toc = time.time()
        epoch_duration = toc - tic
        print(f"Epoch took: {epoch_duration/60.0:.1f} minutes")

        training_metrics.add_epoch({**train_metrics, **val_metrics, 'epoch_duration': epoch_duration})

        epoch_val_loss = val_metrics["val_tot_loss"]
        if epoch_val_loss < best_loss:
            print(f"    loss improved from {np.round(best_loss, 6)} to {np.round(epoch_val_loss, 6)}. saving best model...")
            best_loss = epoch_val_loss
            save_best_model(model, best_model_path)
    training_metrics.end()
    return training_metrics

if __name__ ==  "__main__":
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
    train_flood(train_csv, val_csv, save_dir, model_name, initial_lr,
        batch_size, n_epochs, gpu, checkpoint_path, model_args={
            'from_pretrained':args.from_pretrained
        })
