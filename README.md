# [LayerCAM: Exploring Hierarchical Class Activation Maps for Localization](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf)
To appear at IEEE Transactions on Image Processing 2021  
<img src="https://github.com/PengtaoJiang/LayerCAM/blob/master/layercam.png" width="100%" height="50%">
This paper aims to generate reliable class activation maps from different CNN layers. The class activation maps generated from the shallow layers of CNN tend to capture fine-grained object localization information, while the maps generated from the deep layers tend to generally locate the target objects. 

## Update
Merged into [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (2.9K Stars).

## Run 
```
python test.py --img_path=images/ILSVRC2012_val_00000057.JPEG(or your own image path)
```
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
