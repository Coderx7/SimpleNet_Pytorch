# SimpleNet in Pytorch
SimpleNetV1 architecture implementation in Pytorch 

## Lets Keep it simple, Using simple architectures to outperform deeper and more complex architectures (2016).

![GitHub Logo](https://github.com/Coderx7/SimpleNet/raw/master/SimpNet_V1/images(plots)/SimpleNet_Arch_Larged.jpg)


This is the pytorch implementation of our architecture [SimpleNetV1(2016)](https://arxiv.org/abs/1608.06037) .   
Pytorch is different from caffe in several sections, and this made it a bit harder to have the architecture properly ported especially since I'm a complete newbie in Pytorch. However, thanks to [this great work](https://github.com/D-X-Y/ResNeXt-DenseNet), I could easily focus on the model and port the architecture and hopefully achieve my reported results in Caffe and also exceed them as well! 

The pytorch implementation is also very effieicent and the whole model takes only **1239MB** with the batch size of 64! (compare this to other architectures such as ResNet,WRN, DenseNet which a 800K model takes more than 6G of vram!) 


The original Caffe implementation can be found here : [Original Caffe implementation - 2016](https://github.com/Coderx7/SimpleNet)

#### CIFAR10/100 Results achieved using this implementation :

| Dataset | Accuracy |
|------------|----------|
| CIFAR10    | **95.32**    |
| CIFAR100   | **77.83**   |

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
| SimpleNet                    |    5.48M     |   **95.32**   | **77.83**    |



#### Models and logs  
-- Models and training logs can be found in [snapshot folder](https://github.com/Coderx7/SimpleNet_Pytorch/tree/master/snapshots).



#### How to run ? 
Simply initiate the training like :  
`python3 main.py ./data/cifar.python --dataset cifar10 --arch simplenet --save_path ./snapshots/simplenet --epochs 540 --batch_size 100 --workers 2`

Note that, the initial learning rate, and optimization policy is hard coded just like caffe.

## Citation
If you find SimpleNet useful in your research, please consider citing:

    @article{hasanpour2016lets,
      title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
      author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
      journal={arXiv preprint arXiv:1608.06037},
      year={2016}
    }
