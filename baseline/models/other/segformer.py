import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from .unet import OutConv, Up

class SiameseSegformer(nn.Module):
    def __init__(self, pretrained_model_name_or_path='nvidia/mit-b0', n_classes=5):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path, num_labels=64)
        self.penultimate_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.outc1 = OutConv(64, n_classes)

    def forward_once(self, x):
        return self.segformer(x).logits

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        x = torch.cat([out1, out2], dim=1)
        x = self.penultimate_conv(x)
        x = self.outc1(x)
        # Segformer reduces spatial dimensions by factor of 4, so need to upscale
        x = F.interpolate(x, scale_factor=4)
        return x

# TODO(adrs): Non-siamese version

# https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/segformer#overview
class SiameseSegformer_b0(SiameseSegformer):
    def __init__(self, num_classes=5, num_channels=3):
        super().__init__('nvidia/mit-b0', num_classes)

class SiameseSegformer_b1(SiameseSegformer):
    def __init__(self, num_classes=5, num_channels=3):
        super().__init__('nvidia/mit-b1', num_classes)

# Models larger than b1 do not fit in P6000 GPU memory
