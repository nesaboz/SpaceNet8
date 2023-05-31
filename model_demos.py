import torch
import torchvision
import transformers
from PIL import Image

def get_model(name):
	# TODO: get pre-trained versions of these
	auto_model_configs = {
		# Beit
		# - Needs batch size >1 (because of batch norm?)
		# - Shrinks each spatial dimension by factor of 4
		'beit':transformers.BeitConfig(image_size=size, num_labels=5),
		# DPT
		# - Needs batch size >1 (because of batch norm?)
		# - Shrinks each spatial dimension by factor of 4
        'data2vec-vision': transformers.Data2VecVisionConfig(image_size=size, num_labels=5),
		# DPT
		# - Works with batch size 1
		# - Preserves spatial dimensions
        'dpt': transformers.DPTConfig(image_size=size, num_labels=5),
		# MobileNetV2
		# - Needs batch size >1 (because of batch norm?)
		# - Shrinks each spatial dimension by factor of 32 by default!
		# - Shrinks each spatial dimension by factor of 8 with other settings
        'mobilenet_v2': transformers.MobileNetV2Config(image_size=size, num_labels=5, output_stride=8),
		# MobileViT
		# - Needs batch size >1 (because of batch norm?)
		# - Shrinks each spatial dimension by factor of 32 by default!
		# - Shrinks each spatial dimension by factor of 8 with other settings
        'mobilevit': transformers.MobileViTConfig(image_size=size, num_labels=5, output_stride=8),
		# Segformer
		# - Works with batch size 1
		# - Shrinks each spatial dimension by factor of 4
        'segformer': transformers.SegformerConfig(image_size=size, num_labels=5),
		# UperNet
		# https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/upernet#overview
		# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/UPerNet/Perform_inference_with_UperNetForSemanticSegmentation_(Swin_backbone).ipynb

		# This does not work (yet!)
		# #
		# # - Needs batch size >1 (because of batch norm?)
		# # - RuntimeError: Given groups=1, weight of size [256, 384, 3, 3],
		# #   expected input[2, 1024, 14, 14] to have 384 channels, but got 1024
		# #   channels instead 
        # 'upernet-convnext': transformers.UperNetConfig(
		# 	image_size=size, num_labels=5,
		# 	backbone_config=transformers.ConvNextConfig()),
	}
	if name in auto_model_configs:
		return transformers.AutoModelForSemanticSegmentation.from_config(
				model_configs[model_name])
	if name == 'upernet-swin':
		# - Works with batch size 1
		# - Preserves spatial dimensions
		# There are other sizes to try
		return transformers.UperNetForSemanticSegmentation.from_pretrained(
			"openmmlab/upernet-swin-large",
			num_labels=5, ignore_mismatched_sizes=True)
	if name == 'upernet-convnext':
		# - Works with batch size 1
		# - Preserves spatial dimensions
		return transformers.UperNetForSemanticSegmentation.from_pretrained(
			"openmmlab/upernet-convnext-large",
			num_labels=5, ignore_mismatched_sizes=True)


if __name__ == '__main__':
	image_path = '../data/Germany_Training_Public/PRE-event/10500500C4DD7000_0_15_63.tif'
	image = torchvision.transforms.ToTensor()(Image.open(image_path))

	# OOMs with full size images on my laptop with 16GB RAM
	size = 224
	small_image = image[:, :size, :size]
	model_name = 'upernet-convnext'

	# TODO: create image pre-processors for each model

	# Some models use batch norm and require batch sizes >1 :(
	min_batch_size = 1
	batch = torch.stack([small_image]*min_batch_size, dim=0)
	print(batch.shape)
	model = get_model(model_name)
	out = model(batch).logits
	print(out.shape)
