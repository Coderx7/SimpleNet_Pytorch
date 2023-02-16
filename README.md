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

| **Method**                   | **\#Params** |  **ImageNet** | **ImageNet-Real-Labels**  |
| :--------------------------- | :----------: | :-----------: | :-----------: |  
| SimpleNetV1_imagenet(36.33 MB)   |     9.5m     | 74.17/91.614   | 81.24/94.63   |  
| SimpleNetV1_imagenet(21.9 MB)   |     5.7m     | 71.936/90.3    | 79.12/93.68   |         
| SimpleNetV1_imagenet(12.58 MB)   |     3m       | 68.15/87.762   | 75.66/91.80   |  
| SimpleNetV1_imagenet(5.78 MB)    |     1.5m     | 61.524/83.43   | 69.11/88.10   |   

SimpleNet performs very decently, it outperforms VGGNet, variants of ResNet and MobileNets(1-3)   
and its pretty fast as well! and its all using plain old CNN!.  

Here's an example of benchmark run on small variants of simplenet and some other known architectures such as mobilenets.    
Small variants of simplenet consistently achieve high performance/accuracy:  

|       model                       | samples_per_sec   |  param_count  | top1   | top5   |
|:----------------------------------| :--------------:  | :-----------: | :--:   | :---:  |  
|mobilenetv3_small_050              |     3035.37       | 1.59          | 57.89  | 80.194 |
|**simplenetv1_small_m1_05**            |     2839.35         | 1.51          | **60.89**|**82.978**|
|lcnet_050                          |     2683.57       | 1.88          | 63.1   | 84.382 |
|**simplenetv1_small_m2_05**            |     2340.51       | 1.51          |**61.524**|**83.432**|
|mobilenetv3_small_075              |     1781.14       | 2.04          | 65.242 | 85.438 |
|tf_mobilenetv3_small_075           |     1674.31       | 2.04          | 65.714 | 86.134 |
|**simplenetv1_small_m1_075**           |     1524.64       | 3.29          |**67.764**|**87.66** |
|tf_mobilenetv3_small_minimal_100   |     1308.27       | 2.04          | 62.908 | 84.234 |
|**simplenetv1_small_m2_075**           |     1264.33       | 3.29          |**68.15** |**87.762**|
|mobilenetv3_small_100              |     1263.23       | 2.54          | 67.656 | 87.634 |
|tf_mobilenetv3_small_100           |     1220.08       | 2.54          | 67.924 | 87.664 |
|mnasnet_small                      |     1085.15       | 2.03          | 66.206 | 86.508 |
|mobilenetv2_050                    |     848.38        | 1.97          | 65.942 | 86.082 |
|dla46_c                            |     531.0         | 1.3           | 64.866 | 86.294 |
|dla46x_c                           |     318.32        | 1.07          | 65.97  | 86.98  |
|dla60x_c                           |     298.59        | 1.32          | 67.892 | 88.426 |

and this is a sample for larger models: simplenet variants outperform many newer architecures.  

|            model                  |  samples_per_sec | param_count   |  top1  |  top5  |
|:----------------------------------| :--------------: | :-----------: | :--:   | :---:  |  
| vit_tiny_r_s16_p8_224             |     1882.23      |   6.34        | 71.792 | 90.822 |
| simplenetv1_small_m1_075          |     1516.74      |   3.29        | 67.764 | 87.660 |
| simplenetv1_small_m2_075          |     1260.89      |   3.29        | 68.150 | 87.762 |
| simplenetv1_5m_m1                 |     1107.70      |   5.75        | 71.370 | 90.100 |
| deit_tiny_patch16_224             |      991.41      |   5.72        | 72.172 | 91.114 |
| resnet18                          |      876.92      |  11.69        | 69.744 | 89.082 |
| simplenetv1_5m_m2                 |      835.17      |   5.75        | 71.936 | 90.300 |
| crossvit_9_240                    |      602.13      |   8.55        | 73.960 | 91.968 |
| vit_base_patch32_224_sam          |      571.37      |  88.22        | 73.694 | 91.010 |
| tinynet_b                         |      530.15      |   3.73        | 74.976 | 92.184 |
| resnet26                          |      524.36      |  16.00        | 75.300 | 92.578 |
| tf_mobilenetv3_large_075          |      505.13      |   3.99        | 73.436 | 91.344 |
| resnet34                          |      491.96      |  21.80        | 75.114 | 92.284 |
| regnetx_006                       |      478.41      |   6.20        | 73.860 | 91.672 |
| dla34                             |      472.49      |  15.74        | 74.620 | 92.072 |
| simplenetv1_9m_m1                 |      459.21      |   9.51        | 73.376 | 91.048 |
| repvgg_b0                         |      455.36      |  15.82        | 75.160 | 92.418 |
| ghostnet_100                      |      407.03      |   5.18        | 73.974 | 91.460 |
| tf_mobilenetv3_large_minimal_100  |       406.84     |   3.92        | 72.250 | 90.630 |
| mobilenetv3_large_100             |      402.08      |   5.48        | 75.766 | 92.544 |
| simplenetv1_9m_m2                 |      389.94      |   9.51        | 74.170 | 91.614 |
| tf_mobilenetv3_large_100          |      388.30      |   5.48        | 75.518 | 92.604 |
| mobilenetv2_100                   |      295.68      |   3.50        | 72.970 | 91.020 |
| densenet121                       |      293.94      |   7.98        | 75.584 | 92.652 |
| mnasnet_100                       |      262.25      |   4.38        | 74.658 | 92.112 |
| vgg11                             |      260.38      | 132.86        | 69.028 | 88.626 |
| vgg11_bn                          |      248.92      | 132.87        | 70.360 | 89.802 |
| mobilenetv2_110d                  |      230.80      |   4.52        | 75.038 | 92.184 |
| efficientnet_lite0                |      224.81      |   4.65        | 75.476 | 92.512 |
| tf_efficientnet_lite0             |      219.93      |   4.65        | 74.832 | 92.174 |
| vgg13                             |      154.03      | 133.05        | 69.926 | 89.246 |
| vgg13_bn                          |      144.39      | 133.05        | 71.594 | 90.376 |
| vgg16                             |      123.70      | 138.36        | 71.590 | 90.382 |
| vgg16_bn                          |      117.06      | 138.37        | 73.350 | 91.504 |
| vgg19                             |      103.71      | 143.67        | 72.366 | 90.870 |
| vgg19_bn                          |      98.59       | 143.68        | 74.214 | 91.848 |

Benchmark was done using a GTX1080 on Pytorch 1.11 with fp32, nhwc, batchsize of 256, input size = `224x224x3`.   
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
