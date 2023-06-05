import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from .unet import OutConv, Up

class SiameseSegformer(nn.Module):
    def __init__(self, num_classes=5, pretrained_model_name_or_path=None):
        '''
        Constructs a SiameseSegformer model. If pretrained_model_name_or_path is
        not None, the model is initialized from pretrained weights.
        '''
        super().__init__()
        self.from_pretrained = pretrained_model_name_or_path is not None
        if self.from_pretrained:
            print('Initialize from pretrained weights...')
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name_or_path, num_labels=64)
        else:
            print('Do not initialize from pretrained weights...')
            config = SegformerConfig(num_labels=64)
            self.segformer = SegformerForSemanticSegmentation(config)
        self.penultimate_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.outc1 = OutConv(64, num_classes)

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

class Upscale(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)

class Segformer(nn.Module):
    def __init__(self, num_classes=[1, 8], pretrained_model_name_or_path=None):
        super().__init__()
        '''
        Constructs a Segformer model. If pretrained_model_name_or_path is
        not None, the model is initialized from pretrained weights.
        '''
        num_filters = 64

        self.from_pretrained = pretrained_model_name_or_path is not None
        if self.from_pretrained:
            print('Initialize from pretrained weights...')
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name_or_path, num_labels=num_filters)
        else:
            print('Do not initialize from pretrained weights...')
            config = SegformerConfig(num_labels=num_filters)
            self.segformer = SegformerForSemanticSegmentation(config)
        self.num_classes = num_classes
        assert(isinstance(self.num_classes, int) or isinstance(self.num_classes, list))

        if isinstance(num_classes, int):
            self.final = self.make_final_classifier(num_filters, num_classes)
        else: # num_classes is a list. see assert above. 
            self.final1 = self.make_final_classifier(num_filters, num_classes[0])
            self.final2 = self.make_final_classifier(num_filters, num_classes[1])

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 3, padding=1),
            Upscale(4)
        )

    def make_final_classifier2(self, in_filters, num_classes):
        return nn.Sequential(
                nn.Conv2d(in_filters, 32, 3, padding=1),
                nn.Conv2d(32, num_classes, 3, padding=1),
                Upscale(4))

    def forward(self, x):
        x = self.segformer(x).logits
        if isinstance(self.num_classes, int):
            return self.final(x)
        else:
            f1 = self.final1(x)
            f2 = self.final2(x)
            return f1, f2

# https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/segformer#overview
class Segformer_b0(Segformer):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        pretrained_model_name_or_path = None
        if from_pretrained:
            pretrained_model_name_or_path = 'nvidia/mit-b0'
        super().__init__(num_classes, pretrained_model_name_or_path)

class Segformer_b1(Segformer):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        pretrained_model_name_or_path = None
        if from_pretrained:
            pretrained_model_name_or_path = 'nvidia/mit-b1'
        super().__init__(num_classes, pretrained_model_name_or_path)

class SiameseSegformer_b0(SiameseSegformer):
    def __init__(self, num_classes=5, num_channels=3, from_pretrained=True):
        pretrained_model_name_or_path = None
        if from_pretrained:
            pretrained_model_name_or_path = 'nvidia/mit-b0'
        super().__init__(num_classes, pretrained_model_name_or_path)

class SiameseSegformer_b1(SiameseSegformer):
    def __init__(self, num_classes=5, num_channels=3, from_pretrained=True):
        pretrained_model_name_or_path = None
        if from_pretrained:
            pretrained_model_name_or_path = 'nvidia/mit-b1'
        super().__init__(num_classes, pretrained_model_name_or_path)

# Models larger than b1 do not fit in P6000 GPU memory
