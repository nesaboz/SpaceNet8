import os
import torch
import torch.nn as nn
import numpy as np
from datasets.datasets import SN8Dataset
from models.other.eff_unet import EffUNet
from models.other.unet import UNetSiamese



if __name__ == '__main__':
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    dataset = SN8Dataset('../../../../data/spacenet8/adrs-small-train.csv',
                            data_to_load=["preimg","postimg","flood"])
    celoss = nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model = EffUNet(in_channels = 6, classes = 5)

    
    model.train()
    model.cuda()
    for i, data in enumerate(dataloader):
        preimg, postimg, building, road, roadspeed, flood = data
        combinedimg = torch.cat((preimg, postimg), dim=1)
        combinedimg = combinedimg.cuda().float()

        preimg = preimg.cuda().float()
        postimg = postimg.cuda().float()

        flood = flood.numpy()
        flood_shape = flood.shape
        flood = np.append(np.zeros(shape=(flood_shape[0],1,flood_shape[2],flood_shape[3])), flood, axis=1)
        flood = np.argmax(flood, axis = 1) # this is needed for cross-entropy loss. 
        flood_cropped= flood[:, 26:1274, 26:1274]
        flood_cropped = torch.tensor(flood_cropped).cuda()
        print('Flood shape (cropped):', flood.shape)
        cropped_image = combinedimg[:, :, 26:1274, 26:1274]
        print('input shape (cropped):', cropped_image.shape)
        with torch.no_grad():
            flood_pred = model(cropped_image)
            print('Flood pred shape:', flood_pred.shape)
            loss = celoss(flood_pred, flood_cropped.long())
            print(loss)
        break