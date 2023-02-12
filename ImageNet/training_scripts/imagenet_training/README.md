
## Train, Validation, Inference Scripts
As I mentioned earlier this is a modified version of [timm](https://rwightman.github.io/pytorch-image-models/), and its modified to
include simplenet and whats needed for training and experimenting with it. Therefore its basically the same codebase and everything
in timms documentation applies here.

The root folder of the repository contains reference train, validation, and inference scripts that work with the included models and other features of this repository. They are adaptable for other datasets and use cases with a little hacking. See [documentation](https://rwightman.github.io/pytorch-image-models/scripts/) for some basics and [training hparams](https://rwightman.github.io/pytorch-image-models/training_hparam_examples) for some train examples that produce SOTA ImageNet results.


## Licenses

### training Code
The code here is licensed Apache 2.0. I've taken care to make sure any third party code included or adapted has compatible (permissive) licenses such as MIT, BSD, etc. I've made an effort to avoid any GPL / LGPL conflicts. That said, it is your responsibility to ensure you comply with licenses here and conditions of any dependent licenses. Where applicable, I've linked the sources/references for various components in docstrings. If you think I've missed anything please create an issue.

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

