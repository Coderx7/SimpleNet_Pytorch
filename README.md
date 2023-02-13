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
| SimpleNetV1_imagenet(38 MB)   |     9.5m     | 74.17/91.61   | 81.24/94.63   |  
| SimpleNetV1_imagenet(23 MB)   |     5.7m     | 71.94/90.3    | 79.12/93.68   |         
| SimpleNetV1_imagenet(13 MB)   |     3m       | 68.15/87.76   | 75.66/91.80   |  
| SimpleNetV1_imagenet(6 MB)    |     1.5m     | 61.53/83.43   | 69.11/88.10   |   

SimpleNet performs very decently, it outperforms VGGNet, variants of ResNet and MobileNets(1-3)   
and its fast, very fast!  

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
