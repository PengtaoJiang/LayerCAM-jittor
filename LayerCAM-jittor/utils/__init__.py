#!/usr/bin/env python
"""flashtorch.utils

This module provides utility functions for image handling and tensor
transformation.

"""
import jittor as jt
from jittor import init
from jittor import nn
from PIL import Image
import matplotlib.pyplot as plt
from .imagenet import *
import jittor.transform as transforms


def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def apply_transforms(image, size=224):
    if (not isinstance(image, Image.Image)):
        image = F.to_pil_image(image)
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(), transforms.ImageNormalize(means, stds)])
    tensor = transform(image)
    tensor = jt.array(tensor).unsqueeze(0)
    return tensor

def apply_transforms_v0(image, size=224):
    if (not isinstance(image, Image.Image)):
        image = F.to_pil_image(image)
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad = True
    return tensor

def denormalize(tensor):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    denormalized = tensor.copy()
    for (channel, mean, std) in zip(denormalized[0], means, stds):
        channel * std + mean
    return denormalized

def standardize_and_clip(tensor, min_value=0.0, max_value=1.0):
    mean = tensor.mean()
    std = tensor.std()
    if (std == 0):
        std += 1e-07
    standardized = (tensor - mean) / std * 0.1
    clipped = (standardized + 0.5)
    clipped[clipped > max_value] = max_value
    clipped[clipped < min_value] = min_value
    return clipped

def format_for_plotting(tensor):
    has_batch_dimension = (len(tensor.shape) == 4)
    formatted = tensor.copy()
    if has_batch_dimension:
        formatted = tensor.squeeze(0)
    if (formatted.shape[0] == 1):
        return formatted.squeeze(0)
    else:
        return formatted.transpose((1, 2, 0))

def visualize(input_, gradients, save_path=None, cmap='viridis', alpha=0.7):
    input_ = format_for_plotting(denormalize(input_))
    gradients = format_for_plotting(standardize_and_clip(gradients))
    subplots = [('Input image', [(input_, None, None)]), ('Saliency map across RGB channels', [(gradients, None, None)]), ('Overlay', [(input_, None, None), (gradients, cmap, alpha)])]
    num_subplots = len(subplots)
    fig = plt.figure(figsize=(16, 3))
    for (i, (title, images)) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, (i + 1))
        ax.set_axis_off()
        for (image, cmap, alpha) in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)
        ax.set_title(title)
    if (save_path is not None):
        plt.savefig(save_path)

def basic_visualize(input_, gradients, save_path=None, weight=None, cmap='viridis', alpha=0.7):
    input_ = format_for_plotting(denormalize(input_))
    gradients = format_for_plotting(standardize_and_clip(gradients))
    subplots = [('Saliency map across RGB channels', [(gradients, None, None)]), ('Overlay', [(input_, None, None), (gradients, cmap, alpha)])]
    num_subplots = len(subplots)
    fig = plt.figure(figsize=(4, 4))
    for (i, (title, images)) in enumerate(subplots):
        ax = fig.add_subplot(1, num_subplots, (i + 1))
        ax.set_axis_off()
        for (image, cmap, alpha) in images:
            ax.imshow(image, cmap=cmap, alpha=alpha)
    if (save_path is not None):
        plt.savefig(save_path)

def find_resnet_layer(arch, target_layer_name):
    if (target_layer_name is None):
        target_layer_name = 'layer4'
    if ('layer' in target_layer_name):
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if (layer_num == 1):
            target_layer = arch.layer1
        elif (layer_num == 2):
            target_layer = arch.layer2
        elif (layer_num == 3):
            target_layer = arch.layer3
        elif (layer_num == 4):
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))
        if (len(hierarchy) >= 2):
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]
        if (len(hierarchy) >= 3):
            target_layer = target_layer._modules[hierarchy[2]]
        if (len(hierarchy) == 4):
            target_layer = target_layer._modules[hierarchy[3]]
    else:
        target_layer = arch._modules[target_layer_name]
    return target_layer

def find_densenet_layer(arch, target_layer_name):
    if (target_layer_name is None):
        target_layer_name = 'features'
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]
    if (len(hierarchy) >= 2):
        target_layer = target_layer._modules[hierarchy[1]]
    if (len(hierarchy) >= 3):
        target_layer = target_layer._modules[hierarchy[2]]
    if (len(hierarchy) == 4):
        target_layer = target_layer._modules[hierarchy[3]]
    return target_layer

def find_vgg_layer(arch, target_layer_name):
    if (target_layer_name is None):
        target_layer_name = 'features'
    hierarchy = target_layer_name.split('_')
    if (len(hierarchy) >= 1):
        target_layer = arch.features
    if (len(hierarchy) == 2):
        target_layer = target_layer[int(hierarchy[1])]
    return target_layer

def find_alexnet_layer(arch, target_layer_name):
    if (target_layer_name is None):
        target_layer_name = 'features_29'
    hierarchy = target_layer_name.split('_')
    if (len(hierarchy) >= 1):
        target_layer = arch.features
    if (len(hierarchy) == 2):
        target_layer = target_layer[int(hierarchy[1])]
    return target_layer

def find_squeezenet_layer(arch, target_layer_name):
    if (target_layer_name is None):
        target_layer_name = 'features'
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]
    if (len(hierarchy) >= 2):
        target_layer = target_layer._modules[hierarchy[1]]
    if (len(hierarchy) == 3):
        target_layer = target_layer._modules[hierarchy[2]]
    elif (len(hierarchy) == 4):
        target_layer = target_layer._modules[((hierarchy[2] + '_') + hierarchy[3])]
    return target_layer

def find_googlenet_layer(arch, target_layer_name):
    if (target_layer_name is None):
        target_layer_name = 'features'
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]
    if (len(hierarchy) >= 2):
        target_layer = target_layer._modules[hierarchy[1]]
    if (len(hierarchy) == 3):
        target_layer = target_layer._modules[hierarchy[2]]
    elif (len(hierarchy) == 4):
        target_layer = target_layer._modules[((hierarchy[2] + '_') + hierarchy[3])]
    return target_layer

def find_mobilenet_layer(arch, target_layer_name):
    if (target_layer_name is None):
        target_layer_name = 'features'
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]
    if (len(hierarchy) >= 2):
        target_layer = target_layer._modules[hierarchy[1]]
    if (len(hierarchy) == 3):
        target_layer = target_layer._modules[hierarchy[2]]
    elif (len(hierarchy) == 4):
        target_layer = target_layer._modules[((hierarchy[2] + '_') + hierarchy[3])]
    return target_layer

def find_shufflenet_layer(arch, target_layer_name):
    if (target_layer_name is None):
        target_layer_name = 'features'
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]
    if (len(hierarchy) >= 2):
        target_layer = target_layer._modules[hierarchy[1]]
    if (len(hierarchy) == 3):
        target_layer = target_layer._modules[hierarchy[2]]
    elif (len(hierarchy) == 4):
        target_layer = target_layer._modules[((hierarchy[2] + '_') + hierarchy[3])]
    return target_layer

def find_layer(arch, target_layer_name):
    if (target_layer_name.split('_') not in arch._modules.keys()):
        raise Exception('Invalid target layer name.')
    target_layer = arch._modules[target_layer_name]
    return target_layer