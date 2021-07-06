import jittor as jt
from jittor import init
from jittor import nn
from cam.basecam import *

class LayerCAM(BaseCAM):

    def __init__(self, model_dict, optimizer):
        super().__init__(model_dict, optimizer)

    def execute(self, input, class_idx=None, retain_graph=False):
        (b, c, h, w) = input.shape
        logit = self.model_arch(input)
        
        if (class_idx is None):
            predicted_class = logit.max(dim=1)[(- 1)]
            score = logit[:, logit.max(dim=1)[(- 1)]].squeeze(0)
        else:
            predicted_class = jt.array64([class_idx])
            score = logit[:, class_idx].squeeze()
            
        one_hot_output = jt.zeros((1, logit.shape[(- 1)]))
        one_hot_output[0][predicted_class] = 1
        #self.model_arch.zero_grad()
        
        # logit.forward(gradient=one_hot_output, retain_graph=True)
        self.optimizer.backward(logit[0][predicted_class])
        activations = self.activations['value'].clone().detach()
        gradients = self.gradients['value'].clone().detach()
        (b, k, u, v) = activations.shape

        with jt.no_grad():
            activation_maps = (activations * nn.relu(gradients))
            cam = jt.sum(activation_maps, dim=1).unsqueeze(0)
            cam = nn.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
            (cam_min, cam_max) = (cam.min(), cam.max())
            norm_cam = (cam - cam_min) / (((cam_max - cam_min) + 1e-08)).data
        return norm_cam

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.execute(input, class_idx, retain_graph)