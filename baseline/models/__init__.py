import models.pytorch_zoo.unet as unet
from models.other.unet import UNetSiamese
import models.other.segformer as segformer
from models.other.siamunetdif import SiamUnet_diff
from models.other.siamnestedunet import SNUNet_ECAM
import models.other.eff_pretrained as Densen_EffUnet

from models.other.unet import UNet

flood_models = {
    'resnet34_siamese': unet.Resnet34_siamese_upsample,
    'resnet34': unet.Resnet34_upsample,
    'resnet50_siamese': unet.Resnet50_siamese_upsample,
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
    'segformer_b0_ade_siamese': segformer.SiameseSegformer_b0_ade,
    'segformer_b0_cityscapes_siamese': segformer.SiameseSegformer_b0_cityscapes,
    'segformer_b1_siamese': segformer.SiameseSegformer_b1,
    'segformer_b2_siamese': segformer.SiameseSegformer_b2,
    'effunet_b2_siamese': Densen_EffUnet.EffUnet_b2,
    'effunet_b4_siamese': Densen_EffUnet.EffUnet_b4,
    'dense_121_siamese': Densen_EffUnet.Dense_121,
    'dense_161_siamese': Densen_EffUnet.Dense_161
}

foundation_models = {
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,
    'unet':UNet,
    'segformer_b0': segformer.Segformer_b0,
    'segformer_b0_ade': segformer.Segformer_b0_ade,
    'segformer_b0_cityscapes': segformer.Segformer_b0_cityscapes,
    'segformer_b1': segformer.Segformer_b1,
    'segformer_b2': segformer.Segformer_b2,
    'segformer_b0_1x1_conv': segformer.Segformer_b0_1x1_conv,
    'segformer_b0_double_conv': segformer.Segformer_b0_double_conv,
    'dummy': segformer.DummyModule,
    'effunet_b2': Densen_EffUnet.EffUnet_b2_f,
    'effunet_b4': Densen_EffUnet.EffUnet_b4_f,
    'dense_121': Densen_EffUnet.Dense_121_f,
    'dense_161': Densen_EffUnet.Dense_161_f
}