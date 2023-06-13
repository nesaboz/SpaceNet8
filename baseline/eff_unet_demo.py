# by default the models have pre-trained weight

import torch
from models.other.eff_pretrained import DensenetUnet, EfficientUnet

test_input = torch.rand((1, 6, 1024, 1024))

# EfficientUnet has the fllowing 2 options:

model = EfficientUnet(5, backbone_arch = 'efficientnet-b2')
model = EfficientUnet(5, backbone_arch = 'efficientnet-b4')

# DensenetUnet has the follwing 2 options:
model = DensenetUnet(5, 'densenet121')
model = DensenetUnet(5, 'densenet161') # much larger model than 121
out = model(test_input)
print("out shape", out.shape)