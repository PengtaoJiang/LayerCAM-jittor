# LayerCAM: Exploring Hierarchical Class Activation Maps for Localization
To appear at IEEE Transactions on Image Processing 2021  
<img src="https://github.com/PengtaoJiang/LayerCAM/blob/master/layercam.png" width="100%" height="50%">
This paper aims to generate reliable class activation maps from different cnn layers. 

## Run 
```
python test.py --img_path=images/ILSVRC2012_val_00000057.JPEG(or your own image path)
```


## Application
### 1. industry defect localization 
The industry defects usually have small size and various shapes. The maps generated from the last cnn layers can only 
coarsely locate the defects.
<img src="https://github.com/PengtaoJiang/LayerCAM/blob/master/defect.png" width="100%" height="100%">


## Citation


## License
The source code is free for research and education use only. Any comercial use should get formal permission first.

## Contact
If you have any questions, feel free to contact me via: pt.jiang at mail.nankai.edu.cn
