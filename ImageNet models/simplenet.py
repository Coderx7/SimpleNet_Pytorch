#In the name of God the most compassionate the most merciful
# SimpleNet implementation for imagenet in pytorch 
# Seyyed Hossein Hasanpour
# coderx7@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, classes=1000, scale=1.0, network_idx=0, mode=2, drop_rates={}):
        super(SimpleNet, self).__init__()
       
        self.cfg = {
                    'simplenetv1_imagenet':    [ (64, 1, 0), (128,1,0), (128,1,0), (128,1,0), (128,1,0), (128,1,0), ('p',2,0), (256,1,0), (256,1,0), (256,1,0),(512,1,0),('p',2,0), (2048,1,0,'k1'), (256,1,0,'k1'), (256,1,0)], 
                   } 
        
        self.dropout_rates = drop_rates 
        # 15 is the last layer (including two pooling layers) signifying the dropout
        # for the very last layer to be used after the pooling not prior to it
        self.last_dropout_rate = self.dropout_rates.get(15, 0.0)
        self.strides = {1: {0:2, 1:2, 2:2},
                        2: {0:2, 1:2, 2:1, 3:2},
                        3: {0:2, 1:2, 2:1, 3:1, 4:2},
                        4: {0:2, 1:1, 2:2, 3:1, 4:2, 5:1},
                        5: {0:2, 1:1, 2:2, 3:1, 4:2, 5:1},
                        6: {0:2, 1:2}
                        }

        self.num_classes = classes
        self.scale = scale
        self.networks = ['simplenetv1_imagenet']
        self.network_idx = network_idx
        self.mode = mode
        
        self.features = self._make_layers(scale)
        self.classifier = nn.Linear(round(self.cfg[self.networks[network_idx]][-1][0] * scale), classes)

    def forward(self, x):
        out = self.features(x)
        #Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = F.dropout2d(out, self.last_dropout_rate, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, scale):
        layers = []
        input_channel = 3
        stride_list = self.strides[self.mode]
        for idx, (layer, stride, drpout,*layer_type) in enumerate(self.cfg[self.networks[self.network_idx]]):
            stride = stride_list[idx] if len(stride_list)>idx else stride 
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
            if layer == 'p':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(stride, stride), dilation=(1, 1), ceil_mode=False),
                           nn.Dropout2d(p=dropout_value,inplace=True)]
            else:
                filters = round(layer * scale)
                # TODO: its better to use a dropout of 0 for all layers, 
                # and dont create an exception for cnns followed by a pooling layer
                if dropout_value is None:
                    layers += [nn.Conv2d(input_channel, filters, kernel_size=kernel_size, stride=(stride, stride), padding=(1, 1)),
                               nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(input_channel, filters, kernel_size=kernel_size, stride=(stride, stride), padding=(1, 1)),
                               nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                               nn.ReLU(inplace=True),
                               #!pytorch 1.11.0+cu113 complains when dropout is inplace here!
                               nn.Dropout2d(p=dropout_value,inplace=False)]

                input_channel = filters

        model = nn.Sequential(*layers)
        print(model)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        return model  

if __name__ == '__main__':
     simplenet = SimpleNet(classes=1000, scale=1.0, network_idx=0, mode=2)
     input_dummy = torch.randn(size=(5,3,224,224))
     out = simplenet(input_dummy)
     print(f'output: {out.size()}')
