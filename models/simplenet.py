'''
SimplerNetV1 in Pytorch.

The implementation is basded on : 
https://github.com/D-X-Y/ResNeXt-DenseNet
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class simplenet(nn.Module):
    def __init__(self, classes=10, simpnet_name='simplenet'):
        super(simplenet, self).__init__()
        #print(simpnet_name)
        self.features = self._make_layers() #self._make_layers(cfg[simpnet_name])
        self.classifier = nn.Linear(256, classes)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        # print(own_state.keys())
        # for name, val in own_state:
        # print(name)
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if name not in own_state:
                # print(name)
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print("STATE_DICT: {}".format(name))
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ... Using Initial Params'.format(
                    name, own_state[name].size(), param.size()))

    def forward(self, x):
        #print(x.size())
        out = self.features(x)

        #Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:]) 
        out = F.dropout2d(out, 0.1, training=True)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):

        model = nn.Sequential(
                             nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),


                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),


                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                             nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),



                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),


                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),


                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),



                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),



                             nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),



                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),


                             nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                             nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),



                             nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),


                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Dropout2d(p=0.1),


                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
                             nn.ReLU(inplace=True),

                            )

        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model


    # def _make_layers(self, cfg):
    #     layers = []
    #     in_channels = 3
    #     i=0
    #     for x in cfg:
    #         if x[0] == 'M':
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2),
    #                        nn.Dropout2d(0.1)]

    #         elif x[0] =='M3':
    #         	layers += [nn.MaxPool2d(kernel_size=2, stride=2),
    #                        nn.Dropout2d(0.1)]
    #         else:
    #             kernel_size = [layer[x[0]][0], layer[x[0]][1]]
    #             stride = layer[x[0]][2]
    #             padding = layer[x[0]][3]
                
    #             if x[1] <= 192:
    #                 drpout_ratio = nn.Dropout2d(0.1)
    #             else:
    #                 drpout_ratio = nn.Dropout2d(0.1)

    #             if (i in (4,9,12)):
    #                 layers += [nn.Conv2d(in_channels, x[1], kernel_size, padding=padding, stride=stride),
    #                            nn.BatchNorm2d(x[1], eps=1e-05, momentum=0.05),
    #                            nn.ReLU(inplace=True)]
    #                 # if (i ==12):
    #                 # lastlayer = layers[-1]
    #                 # layers += [nn.MaxPool2d(kernel_size=lastlayer.size()[2:]),
    #                 #        nn.Dropout2d(0.3)]           
    #             else:
    #                 #print('drp here!',i)
    #                 layers += [nn.Conv2d(in_channels, x[1], kernel_size, padding=padding, stride=stride),
    #                            nn.BatchNorm2d(x[1], eps=1e-05, momentum=0.05),
    #                            nn.ReLU(inplace=True)]
    #                            #,drpout_ratio]

    #             in_channels = x[1]
    #             i =i+1

    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
		  #       # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		  #       # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
		  #   # elif isinstance(m, nn.BatchNorm2d):
		  #   #     m.weight.data.fill_(1)
		  #   #     m.bias.data.zero_()
    #     return nn.Sequential(*layers)


