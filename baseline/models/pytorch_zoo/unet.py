import torch.nn as nn
from .abstract_model import EncoderDecoder, SiameseEncoderDecoder

class SiameseResnet(SiameseEncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name, from_pretrained=True):
        #self.num_channels = num_channels
        super().__init__(num_classes, num_channels, encoder_name, from_pretrained)
        #print ("unet.py, class Resnet, self.num_channels", num_channels)
        #print ("unet.py, class Resnet, EncoderDecoder.num_channels", EncoderDecoder.num_channels)
    
    def get_encoder(self, encoder, layer, num_channels=3):        
        #print ("unet.py, encoder:", encoder)
        #print ("unet.py, encoder.num_channels:", encoder.num_channels)
        if layer == 0:
            #print ("unet.py, encoder.conv1:", encoder.conv1)
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

class Resnet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name, from_pretrained=True):
        #self.num_channels = num_channels
        super().__init__(num_classes, num_channels, encoder_name, from_pretrained)
        #print ("unet.py, class Resnet, self.num_channels", num_channels)
        #print ("unet.py, class Resnet, EncoderDecoder.num_channels", EncoderDecoder.num_channels)
    
    def get_encoder(self, encoder, layer, num_channels=3):        
        #print ("unet.py, encoder:", encoder)
        #print ("unet.py, encoder.num_channels:", encoder.num_channels)
        if layer == 0:
            #print ("unet.py, encoder.conv1:", encoder.conv1)
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

class Resnet34_siamese_upsample(SiameseResnet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels, encoder_name='resnet34',
            from_pretrained=from_pretrained)

class Resnet34_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels, encoder_name='resnet34',
            from_pretrained=from_pretrained)


class Resnet50_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels, encoder_name='resnet50',
            from_pretrained=from_pretrained)


class Resnet101_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels, encoder_name='resnet101',
            from_pretrained=from_pretrained)



#################################
# adapted from XD_XD SN5 senet.py

class Senet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name,
            from_pretrained=True):
        super().__init__(num_classes, num_channels, encoder_name,
            from_pretrained=from_pretrained)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            inplanes = encoder.inplanes
            input_3x3 = encoder.input_3x3

            if input_3x3:
                return nn.Sequential(
                    encoder.layer0.conv1,
                    encoder.layer0.bn1,
                    encoder.layer0.relu1,
                    encoder.layer0.conv2,
                    encoder.layer0.bn2,
                    encoder.layer0.relu2,
                    encoder.layer0.conv3,
                    encoder.layer0.bn3,
                    encoder.layer0.relu3,
                )
            else:
                return nn.Sequential(
                    encoder.layer0.conv1,
                    encoder.layer0.bn1,
                    encoder.layer0.relu1,
                )
        elif layer == 1:
            return nn.Sequential(
                encoder.layer0.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class SeResnet50_upsample(Senet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels,
            encoder_name='se_resnet50', from_pretrained=from_pretrained)


class SeResnet101_upsample(Senet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels,
            encoder_name='se_resnet101', from_pretrained=from_pretrained)


class SeResnet152_upsample(Senet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels,
            encoder_name='se_resnet152', from_pretrained=from_pretrained)


class SeResnext50_32x4d_upsample(Senet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels,
            encoder_name='se_resnext50_32x4d', from_pretrained=from_pretrained)


class SeResnext101_32x4d_upsample(Senet):
    def __init__(self, num_classes, num_channels=3, from_pretrained=True):
        super().__init__(num_classes, num_channels,
            encoder_name='se_resnext101_32x4d', from_pretrained=from_pretrained)
