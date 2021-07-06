import jittor as jt
from jittor import init
from jittor import nn
from jittor import models
import argparse
from utils import *
from cam.layercam import *

jt.flags.use_cuda = 1


def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of LayerCAM')
    parser.add_argument('--img_path', type=str, default='images/ILSVRC2012_val_00000476.JPEG', help='Path of test image')
    parser.add_argument('--layer_id', type=list, default=[4, 9, 16, 23, 30], help='The cam generation layer')
    return parser.parse_args()
    
if (__name__ == '__main__'):
    args = get_arguments()
    input_image = load_image(args.img_path)
    input_ = apply_transforms(input_image)

    vgg = models.vgg16(pretrained=True)
    optimizer = nn.SGD(vgg.parameters(), 0.1)


    for i in range(len(args.layer_id)):
        layer_name = ('features_' + str(args.layer_id[i]))
        vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name=layer_name, input_size=(224, 224))
        vgg_layercam = LayerCAM(vgg_model_dict, optimizer)
        predicted_class = vgg(input_).max(1)[(- 1)]
        layercam_map = vgg_layercam(input_)
        basic_visualize(input_.numpy(), layercam_map.numpy(), save_path='./vis/stage_{}.png'.format((i + 1)))