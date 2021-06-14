# pip install importlib_resources

import torch
import torch.nn.functional as F
import torchvision.models as models
import argparse

from utils import *
from cam.layercam import *

def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of LayerCAM')
    parser.add_argument("--img_path", type=str, default='images/ILSVRC2012_val_00000476.JPEG', help='Path of test image')
    parser.add_argument("--layer_id", type=list, default=[4,9,16,23,30], help='The cam generation layer')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    input_image = load_image(args.img_path)
    input_ = apply_transforms(input_image)
    if torch.cuda.is_available():
      input_ = input_.cuda()

    vgg = models.vgg16(pretrained=True).eval()
    for i in range(len(args.layer_id)):
        layer_name = 'features_' + str(args.layer_id[i])
        vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name=layer_name, input_size=(224, 224))
        vgg_layercam = LayerCAM(vgg_model_dict)
        predicted_class = vgg(input_).max(1)[-1]

        layercam_map = vgg_layercam(input_)
        basic_visualize(input_.cpu().detach(), layercam_map.type(torch.FloatTensor).cpu(),save_path='./vis/stage_{}.png'.format(i+1))
