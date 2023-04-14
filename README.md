بسم الله الرحمن الرحیم  
پیاده سازی پایتورچ سیمپل نت
# SimpleNet in Pytorch
SimpleNetV1 architecture implementation in Pytorch 

## Lets Keep it simple, Using simple architectures to outperform deeper and more complex architectures (2016).

![GitHub Logo](https://github.com/Coderx7/SimpleNet/raw/master/SimpNet_V1/images(plots)/SimpleNet_Arch_Larged.jpg)


This is the pytorch implementation of our architecture [SimpleNetV1(2016)](https://arxiv.org/abs/1608.06037) .   
Pytorch is different from caffe in several sections, and this made it a bit harder to have the architecture properly ported especially since I'm a complete newbie in Pytorch. However, thanks to [this great work](https://github.com/D-X-Y/ResNeXt-DenseNet), I could easily focus on the model and port the architecture and hopefully achieve my reported results in Caffe and also exceed them as well! 

The pytorch implementation is also very effieicent and the whole model takes only **1239MB** with the batch size of 64! (compare this to other architectures such as ResNet,WRN, DenseNet which a 800K model takes more than 6G of vram!)   



#### Update History:  

<pre>
-- 2023 Apr 14:
  -- update benchmark results
-- 2023 Apr 13:
  -- new weights for the removed paddings for 1x1 conv layers.
  -- some minor fixes
-- 2023 Feb 12: 
  -- re-structured the repository, moving the old implementation into new directory named 'Cifar` and imagenet into its respective directory
  -- updated the old implementation to work with latest version of pytorch.
  -- updated the imagenet scripts/models compatible with timm and a separate version for pure pytorch uscases
  -- updated pretrained models with the latest results 
-- 2022: Adding initial ImageNet models   
-- 2018: Initial Pytorch implementation (for CIFAR10/100/MNIST/SVHN datasets)    
-- 2016: Initial model release for caffe
</pre>


The original Caffe implementation can be found here : [Original Caffe implementation - 2016](https://github.com/Coderx7/SimpleNet)     
  
#### ImageNet Result:  

|       **Model**                  | **\#Params** |  **ImageNet**   | **ImageNet-Real-Labels**  |
| :---------------------------     | :----------: | :-----------:   |   :------------------:    |  
| simplenetv1_9m_m2(36.3 MB)       |     9.5m     | 74.23 / 91.748  |      81.22 / 94.756         |  
| simplenetv1_5m_m2(22 MB)         |     5.7m     | 72.03 / 90.324  |      79.328/ 93.714        |         
| simplenetv1_small_m2_075(12.6 MB)|     3m       | 68.506/ 88.15   |      76.283/ 92.02         |  
| simplenetv1_small_m2_05(5.78 MB) |     1.5m     | 61.67 / 83.488  |      69.31 / 88.195        |  

SimpleNet performs very decently, it outperforms VGGNet, variants of ResNet and MobileNets(1-3)   
and its pretty fast as well! and its all using plain old CNN!.  

Here's an example of benchmark run on small variants of simplenet and some other known architectures such as mobilenets.    
Small variants of simplenet consistently achieve high performance/accuracy:  

|       model                       | samples_per_sec   |  param_count  | top1   | top5   |
|:----------------------------------| :--------------:  | :-----------: | :--:   | :---:  |  
|**simplenetv1_small_m1_05**        |     3100.26       | **1.51**      | **61.122** | **82.988** |
|mobilenetv3_small_050              |     3082.85       | 1.59          | 57.89  | 80.194 |
|lcnet_050                          |     2713.02       | 1.88          | 63.1   | 84.382 |
|**simplenetv1_small_m2_05**        |     2536.16       | **1.51**      | **61.67** | **83.488** |
|mobilenetv3_small_075              |     1793.42       | 2.04          | 65.242 | 85.438 |
|tf_mobilenetv3_small_075           |     1689.53       | 2.04          | 65.714 | 86.134 |
|**simplenetv1_small_m1_075**       |     1626.87       | **3.29**      | **67.784** | **87.718** |
|tf_mobilenetv3_small_minimal_100   |     1316.91       | 2.04          | 62.908 | 84.234 |
|**simplenetv1_small_m2_075**       |     1313.6        | **3.29**      | **68.506** | **88.15**  |
|mobilenetv3_small_100              |     1261.09       | 2.54          | 67.656 | 87.634 |
|tf_mobilenetv3_small_100           |     1213.03       | 2.54          | 67.924 | 87.664 |
|mnasnet_small                      |     1089.33       | 2.03          | 66.206 | 86.508 |
|mobilenetv2_050                    |     857.66        | 1.97          | 65.942 | 86.082 |
|dla46_c                            |     537.08        | 1.3           | 64.866 | 86.294 |
|dla46x_c                           |     323.03        | 1.07          | 65.97  | 86.98  |
|dla60x_c                           |     301.71        | 1.32          | 67.892 | 88.426 |

and this is a sample for larger models: simplenet variants outperform many newer architecures.  

|               model               |  samples_per_sec |  param_count  |  top1  |  top5   |
|:----------------------------------| :--------------: | :-----------: | :----: | :----:  |  
| **simplenetv1_small_m1_075**      |     2893.91      |     **3.29**  | **67.784** | **87.718**  |
| **simplenetv1_small_m2_075**      |     2478.41      |     **3.29**  | **68.506** | **88.15**   |
| vit_tiny_r_s16_p8_224             |     2337.23      |     6.34      | 71.792 | 90.822  |
| **simplenetv1_5m_m1**             |     2105.06      |     **5.75**  | **71.548** | **89.94**   |
| **simplenetv1_5m_m2**             |     1754.25      |     **5.75**  | **72.03**  | **90.324**  |
| resnet18                          |     1750.38      |     11.69     | 69.744 | 89.082  |
| regnetx_006                       |     1620.25      |     6.2       | 73.86  | 91.672  |
| mobilenetv3_large_100             |     1491.86      |     5.48      | 75.766 | 92.544  |
| tf_mobilenetv3_large_minimal_100  |     1476.29      |     3.92      | 72.25  | 90.63   |
| tf_mobilenetv3_large_075          |     1474.77      |     3.99      | 73.436 | 91.344  |
| ghostnet_100                      |     1390.19      |     5.18      | 73.974 | 91.46   |
| tinynet_b                         |     1345.82      |     3.73      | 74.976 | 92.184  |
| tf_mobilenetv3_large_100          |     1325.06      |     5.48      | 75.518 | 92.604  |
| mnasnet_100                       |     1183.69      |     4.38      | 74.658 | 92.112  |
| mobilenetv2_100                   |     1101.58      |     3.5       | 72.97  | 91.02   |
| **simplenetv1_9m_m1**             |     1048.91      |     **9.51**  | **73.792** | **91.486**  |
| resnet34                          |     1030.4       |     21.8      | 75.114 | 92.284  |
| deit_tiny_patch16_224             |     990.85       |     5.72      | 72.172 | 91.114  |
| efficientnet_lite0                |     977.76       |     4.65      | 75.476 | 92.512  |
| **simplenetv1_9m_m2**             |     900.45       |     **9.51**  | **74.23**  | **91.748**  |
| tf_efficientnet_lite0             |     876.66       |     4.65      | 74.832 | 92.174  |
| dla34                             |     834.35       |     15.74     | 74.62  | 92.072  |
| mobilenetv2_110d                  |     824.4        |     4.52      | 75.038 | 92.184  |
| resnet26                          |     771.1        |     16        | 75.3   | 92.578  |
| repvgg_b0                         |     751.01       |     15.82     | 75.16  | 92.418  |
| crossvit_9_240                    |     606.2        |     8.55      | 73.96  | 91.968  |
| vgg11                             |     576.32       |     132.86    | 69.028 | 88.626  |
| vit_base_patch32_224_sam          |     561.99       |     88.22     | 73.694 | 91.01   |
| vgg11_bn                          |     504.29       |     132.87    | 70.36  | 89.802  |
| densenet121                       |     435.3        |     7.98      | 75.584 | 92.652  |
| vgg13                             |     363.69       |     133.05    | 69.926 | 89.246  |
| vgg13_bn                          |     315.85       |     133.05    | 71.594 | 90.376  |
| vgg16                             |     302.84       |     138.36    | 71.59  | 90.382  |
| vgg16_bn                          |     265.99       |     138.37    | 73.35  | 91.504  |
| vgg19                             |     259.82       |     143.67    | 72.366 | 90.87   |
| vgg19_bn                          |     229.77       |     143.68    | 74.214 | 91.848  |

Benchmark was done using a GTX1080 on Pytorch 1.11 with fp32, nchw, batchsize of 256, input size = `224x224x3`.   
For all benchmark results [look here](https://github.com/Coderx7/SimpleNet_Pytorch/tree/master/ImageNet/training_scripts/imagenet_training/results) 

-- The models pretrained weights (pytorch, onnx, jit) can be found in [Release section](https://github.com/Coderx7/SimpleNet_Pytorch/releases)  


#### CIFAR10/100 Results achieved using this implementation :

| Dataset | Accuracy |
|------------|----------|
| CIFAR10    | **95.51**    |
| CIFAR100   | **78.37**   |

### CIFAR10/100 top results (2016): 

| **Method**                   | **\#Params** |  **CIFAR10**  | **CIFAR100** |
| :--------------------------- | :----------: | :-----------: | :----------: |
| VGGNet(16L) /Enhanced        |     138m     | 91.4 / 92.45  |      \-      |
| ResNet-110L / 1202L  \*      |  1.7/10.2m   | 93.57 / 92.07 | 74.84/72.18  |
| SD-110L / 1202L              |  1.7/10.2m   | 94.77 / 95.09 |  75.42 / -   |
| WRN-(16/8)/(28/10)           |    11/36m    | 95.19 / 95.83 |  77.11/79.5  |
| Highway Network              |     N/A      |     92.40     |    67.76     |
| FitNet                       |      1M      |     91.61     |    64.96     |
| FMP\* (1 tests)              |     12M      |     95.50     |    73.61     |
| Max-out(k=2)                 |      6M      |     90.62     |    65.46     |
| Network in Network           |      1M      |     91.19     |    64.32     |
| DSN                          |      1M      |     92.03     |    65.43     |
| Max-out NIN                  |      \-      |     93.25     |    71.14     |
| LSUV                         |     N/A      |     94.16     |     N/A      |
| SimpleNet                    |    5.48M     |   **95.51**   | **78.37**    |



#### Models and logs  
-- refer to each dataset directory in the repository for further information on how to access models.


## Citation
If you find SimpleNet useful in your research, please consider citing:

    @article{hasanpour2016lets,
      title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
      author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
      journal={arXiv preprint arXiv:1608.06037},
      year={2016}
    }
