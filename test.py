# pip install importlib_resources

import torch
import torch.nn.functional as F
import torchvision.models as models

from utils import *
from cam.layercam import *

# alexnet
alexnet = models.alexnet(pretrained=True).eval()
alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_10',input_size=(224, 224))
alexnet_layercam = LayerCAM(alexnet_model_dict)

input_image = load_image('images/'+'ILSVRC2012_val_00002193.JPEG')
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = alexnet(input_).max(1)[-1]

layercam_map = alexnet_layercam(input_)
basic_visualize(input_.cpu().detach(), layercam_map.type(torch.FloatTensor).cpu(),save_path='alexnet.png')

# vgg
vgg = models.vgg16(pretrained=True).eval()
vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_30',input_size=(224, 224))
vgg_layercam = LayerCAM(vgg_model_dict)

input_image = load_image('images/'+'ILSVRC2012_val_00002193.JPEG')
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = vgg(input_).max(1)[-1]

layercam_map = vgg_layercam(input_)
basic_visualize(input_.cpu().detach(), layercam_map.type(torch.FloatTensor).cpu(),save_path='vgg.png')

# resnet
resnet = models.resnet18(pretrained=True).eval()
resnet_model_dict = dict(type='resnet18', arch=resnet, layer_name='layer4',input_size=(224, 224))
resnet_layercam = LayerCAM(resnet_model_dict)

input_image = load_image('images/'+'ILSVRC2012_val_00002193.JPEG')
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = resnet(input_).max(1)[-1]

layercam_map = resnet_layercam(input_)
basic_visualize(input_.cpu().detach(), layercam_map.type(torch.FloatTensor).cpu(),save_path='resnet.png')
