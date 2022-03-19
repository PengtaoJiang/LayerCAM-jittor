# Jittor and PyTorch implementation of [LayerCAM: Exploring Hierarchical Class Activation Maps for Localization](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf)
To appear at IEEE Transactions on Image Processing 2021  
<img src="https://github.com/PengtaoJiang/LayerCAM/blob/master/layercam.png" width="100%" height="50%">
This paper aims to generate reliable class activation maps from different CNN layers. The class activation maps generated from the shallow layers of CNN tend to capture fine-grained object localization information, while the maps generated from the deep layers tend to generally locate the target objects. 

## Update
**`2022.3.19`**: The localization code is released [layercam_loc](https://github.com/PengtaoJiang/layercam_loc).  
**`2021.10.31`**: A simple colab tutorial implemented by frgfm [frgfm/notebooks](https://github.com/frgfm/notebooks).  
**`2021.8.13`**: Merged into [keisen/tf-keras-vis](https://github.com/keisen/tf-keras-vis).  
**`2021.7.14`**: Merged into [frgfm/torch-cam](https://github.com/frgfm/torch-cam).  
**`2021.7.12`**: Merged into [utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) (5.8K Stars).  
**`2021.7.10`**: Merged into [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (2.9K Stars).


## Run 
```
python test.py --img_path=images/ILSVRC2012_val_00000057.JPEG(or your own image path)
```

## Some issues
For those layers that are followed by a layer (max pooling in vgg or conv with stride > 1 in resnet), the cam visualizations usually have grid effect. 
This issue comes from the gradient backward. There are two ways to avoid this issue.   
1. I usually choose the nearby layers for visualization. For example, pool4 instead of conv3_3 in vgg16, or model.layer3[-2] instead of model.layer3[-1] in ResNet. 
2. Another choice is to upsample the following layer's gradient, for example, Up(pool4's gradient) * conv3_3's activation. 
3. Besides, we also found that larger input will obtain more fine-grained cam visualization for lower layers.

## Citation
```
@article{jiang2021layercam,
  title={LayerCAM: Exploring Hierarchical Class Activation Maps For Localization},
  author={Jiang, Peng-Tao and Zhang, Chang-Bin and Hou, Qibin and Cheng, Ming-Ming and Wei, Yunchao},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```

## Contact
If you have any questions, feel free to contact me via: pt.jiang at mail.nankai.edu.cn

## Note
Thanks to [Haofan Wang](https://github.com/haofanwang/Score-CAM). The format of this code is borrowed from [Score-CAM](https://github.com/haofanwang/Score-CAM).
