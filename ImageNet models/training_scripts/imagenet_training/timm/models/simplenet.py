# In the name of God the most compassionate the most merciful
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .registry import register_model


__all__ = [ "simplenet",
            'smpnetv1_05m_m1' , 'smpnetv1_05m_m2',  #1.5m
            'smpnetv1_075m_m1', 'smpnetv1_075m_m2', #3m
            'smpnetv1_5m_m1'  , 'smpnetv1_5m_m2',   #5m
          ]  # model_registry will add each entrypoint fn to this

class SimpleNet(nn.Module):
    def __init__(self, classes:int=1000, scale:float=1.0, network_idx:int=0, mode:int=2, simpnet_name:str="simplenet_imagenet", pretrained:bool=False, drop_rates:dict={},):
        super(SimpleNet, self).__init__()
        # (cnn channels/layer type, stride=1, drp=0.0)
        self.cfg = {
            "simplenetv1_imagenet": [
                (64, 1, 0),
                (128, 1, 0),
                (128, 1, 0),
                (128, 1, 0),
                (128, 1, 0),
                (128, 1, 0),
                ("p", 2, 0),
                (256, 1, 0),
                (256, 1, 0),
                (256, 1, 0),
                (512, 1, 0),
                ("p", 2, 0),
                (2048, 1, 0, "k1"),
                (256, 1, 0, "k1"),
                (256, 1, 0),
            ],
        }

        self.pretrained = pretrained
        self.dropout_rates = drop_rates
        # 15 is the last layer (including two pooling layers) signifying the dropout
        # for the very last layer to be used after the pooling not prior to it
        self.last_dropout_rate = self.dropout_rates.get(15, 0.0)
        self.strides = {
            0: {},
            1: {0: 2, 1: 2, 2: 2},
            2: {0: 2, 1: 2, 2: 1, 3: 2},
            3: {0: 2, 1: 2, 2: 1, 3: 1, 4: 2},
            4: {0: 2, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1},
            5: {0: 2, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1},
            6: {0: 2, 1: 2},
        }

        self.num_classes = classes
        self.scale = scale
        self.networks = ["simplenetv1_imagenet",]
        self.network_idx = network_idx
        self.mode = mode

        self.features = self._make_layers(scale)
        self.classifier = nn.Linear(
            round(self.cfg[self.networks[network_idx]][-1][0] * scale), classes
        )

        if self.pretrained:
            print("load the pretrained model weights!")
            raise Exception("Model weights does not exist!")

    def forward(self, x):
        out = self.features(x)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = F.dropout2d(out, self.last_dropout_rate, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, scale):
        layers = []
        input_channel = 3
        stride_list = self.strides[self.mode]
        for idx, (layer, stride, drpout, *layer_type) in enumerate(
            self.cfg[self.networks[self.network_idx]]
        ):
            stride = stride_list[idx] if len(stride_list) > idx else stride
            # check if any custom dropout_rates are specified for this layer
            # remember pooling also counts as 1 layer, so when creating dropout_rates,
            # consider pooling layer index as well
            # or else, you'll dropout rates will not be at the right layer index!
            dropout_value = self.dropout_rates.get(idx, None)
            dropout_value = drpout if dropout_value is None else dropout_value

            kernel_size = 3
            if layer_type == []:
                kernel_size = 3
            else:
                kernel_size = 1

            # layer either contains the cnn filter count or the letter 'p' signifying a pooling layer
            if layer == "p":
                layers += [nn.MaxPool2d(kernel_size=(2, 2),stride=(stride, stride),dilation=(1, 1),ceil_mode=False,),
                           nn.Dropout2d(p=dropout_value, inplace=True),
                          ]
            else:
                filters = round(layer * scale)
                # TODO: its better to use a dropout of 0 for all layers,
                # and dont create an exception for cnns followed by a pooling layer
                if dropout_value is None:
                    layers += [nn.Conv2d(input_channel,filters,kernel_size=kernel_size,stride=(stride, stride),padding=(1, 1),),
                               nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                               nn.ReLU(inplace=True),]
                else:
                    layers += [nn.Conv2d(input_channel,filters,kernel_size=kernel_size,stride=(stride, stride),padding=(1, 1),),
                               nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                               nn.ReLU(inplace=True),
                               #!pytorch 1.11.0+cu113 complains when dropout is inplace here!
                               nn.Dropout2d(p=dropout_value, inplace=False),]

                input_channel = filters

        model = nn.Sequential(*layers)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        return model

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        # this treats BN layers as separate groups for bn variants, a lot of effort to fix that
        return dict(stem=r"^features\.0", blocks=r"^features\.(\d+)")

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, "gradient checkpointing not supported"

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, network_idx=0, scale=1.0):
        self.num_classes = num_classes
        self.classifier = nn.Linear(round(self.cfg[self.networks[network_idx]][-1][1] * scale), num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = F.dropout2d(x, self.last_dropout_rate, training=self.training)
        x = x.view(x.size(0), -1)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        x = self.forward_features(x)
        return x if pre_logits else self.classifier(x)


def _create_simplenet(pretrained_cfg: str = "simpnet", **kwargs: Any) -> SimpleNet:
    classes = kwargs.get("num_classes", 1000)
    scale = kwargs.get("scale", 1.0)
    netidx = kwargs.get("network_idx", 0)
    mode = kwargs.get("mode", 2)
    rates = kwargs.get("drop_rates", {})
    pretrained = kwargs.get("pretrained", False)
    model = SimpleNet(classes, scale=scale, network_idx=netidx, mode=mode, pretrained=pretrained, drop_rates=rates,)
    return model


@register_model
def simplenet(pretrained_cfg: str = "simplenet", pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    return _create_simplenet(pretrained=pretrained, **kwargs)

@register_model
def smpnetv1_05m_m1(pretrained_cfg: str = 'smpnetv1_05m_m1', **kwargs: Any) -> SimpleNet:
    return _create_simplenet(scale=0.5, network_idx=0, mode=1, **kwargs)

@register_model
def smpnetv1_05m_m2(pretrained_cfg: str = 'smpnetv1_05m_m2', **kwargs: Any) -> SimpleNet:
    return _create_simplenet(scale=0.5, network_idx=0, mode=2, **kwargs)

@register_model
def smpnetv1_075m_m1(pretrained_cfg: str = 'smpnetv1_075m_m1', **kwargs: Any) -> SimpleNet:
    return _create_simplenet(scale=0.75, network_idx=0, mode=1, **kwargs)

@register_model
def smpnetv1_075m_m2(pretrained_cfg: str = 'smpnetv1_075m_m2', **kwargs: Any) -> SimpleNet:
    return _create_simplenet(scale=0.75, network_idx=0, mode=2, **kwargs)

@register_model
def smpnetv1_5m_m1(pretrained_cfg: str = "smpnetv1_5m_m1", **kwargs: Any) -> SimpleNet:
    return _create_simplenet(scale=1.0, network_idx=0, mode=1, **kwargs)

@register_model
def smpnetv1_5m_m2(pretrained_cfg: str = "smpnetv1_5m_m2", **kwargs: Any) -> SimpleNet:
    return _create_simplenet(scale=1.0, network_idx=0, mode=2, **kwargs)


if __name__ == "__main__":
    simplenet = SimpleNet(classes=1000, scale=1.0, network_idx=0, mode=2, simpnet_name="simpnet_imgnet_drpall",)
    input_dummy = torch.randn(size=(5, 224, 224, 3))
    out = simplenet(input_dummy)
    print(f"output: {out.size()}")

