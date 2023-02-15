
## Train, Validation, Inference Scripts
As I mentioned earlier this is a modified version of [timm](https://rwightman.github.io/pytorch-image-models/), and its modified to
include simplenet and whats needed for training and experimenting with it. Therefore its basically the same codebase and everything
in timms documentation applies here.

The root folder of the repository contains reference train, validation, and inference scripts that work with the included models and other features of this repository. They are adaptable for other datasets and use cases with a little hacking. See [documentation](https://rwightman.github.io/pytorch-image-models/scripts/) for some basics and [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) for some train examples that produce SOTA ImageNet results.


## Licenses

### training Code
The code here is licensed Apache 2.0. I've taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc. I've made an effort to avoid any GPL / LGPL conflicts. That said, it is your responsibility to ensure you comply with licenses here and conditions of any dependent licenses. Where applicable, I've linked the sources/references for various components in docstrings. If you think I've missed anything please create an issue.

### How to train the models:
All of the simplenet variants have been trained in the same way, using the same basic training regime, the only difference between them are just:

  1.different weight decay   
  2.different dropout rates or   
  3.disabling label smoothing  
  
Since I had access to a single GPU, and training models take a huge amount of time, I had to come up with a plan to shorten that time.       
to this end, I first train a variant without any dropout, with a specified weight decay, then periodocally save checkpoints (aside from the top10 best checkpoints) so that when a model platued, I can resume from a recent checkpoint with dropout, or a slightly different weight decay. this is not ideal but works well enough when you have no access to decent hardware.   

The models are trained like this:    
e.g lets try to train the simplenetv1_5m_m2 variant, we start by:  
```
./distributed_train.sh 1 /media/hossein/SSD_IMG/ImageNet_DataSet/ --model simplenet --netidx 0 --netscale 1.0 --netmode 2 -b 256 --sched step --epochs 900 --decay-epochs 1 --decay-rate 0.981 --opt rmsproptf --opt-eps .001 -j 20 --warmup-lr 1e-3 --weight-decay 0.00003 --drop 0.0 --amp --lr .0195 --pin-mem --channels-last --model-ema --model-ema-decay 0.9999 
```
then when you see signs of overfitting, you resume from a recent checkpoint with dropout e.g. (for this case I resumed from epoch 251) with these changes(slightly lowered weight decay, added dropout and removed label smoothing):    
```
./distributed_train.sh 1 /media/hossein/SSD_IMG/ImageNet_DataSet/ --model simplenet --netidx 0 --netscale 1.0 --netmode 2 -b 256 --sched step --epochs 900 --decay-epochs 1 --decay-rate 0.981 --opt rmsproptf --opt-eps .002 -j 20 --warmup-lr 1e-3 --weight-decay 0.00002 --drop 0.0 --amp --lr .0195 --pin-mem --channels-last --model-ema --model-ema-decay 0.9999 --resume output/train/20221204-092911-simpnet-224_simplenetv1_2_netmode2_wd3e-5/checkpoint-251.pth\ \(copy\).tar --drop-rates '{"11":0.02,"12":0.05,"13":0.05}' --smoothing 0.0 
```
then we take the average of some of the best checkpoints so far, and if we are not satisfied, we can resume with the average weights and train more. we achieved 71.936 this way.  

Final notes:  

-- All variants are trained using the same training regime with batch-size of 256 on a single GPU.    
-- The small variants such as 1.5m ones, are trained with weight decay of 1e-5, larger models usually use 2e-5 or 3e-5 depending on whether dropout is used or not.  

With decent hardware one should hopefully be able to achieve higher accuracy. If time permits I'll try to improve upon the results inshaallah.  

## Citing

### BibTeX

```bibtex
@article{hasanpour2016lets,
  title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
  author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
  journal={arXiv preprint arXiv:1608.06037},
  year={2016}
}
```

```bibtex
@misc{simplenet_pytorch,
  author = {Seyyed Hossein Hasanpour},
  title = {SimpleNet implementation in Pytorch},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {https://orcid.org/0000-0002-3561-1958},
  howpublished = {\url{https://github.com/Coderx7/SimpleNet_Pytorch}}
}
```

```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

