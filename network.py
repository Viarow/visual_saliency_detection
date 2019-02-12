import torch
import torch.nn as nn

vgg16_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def vgg16(feature_list, input_channels=3, batch_norm=False):
    in_channels = input_channels
    layers = []
    for feat in feature_list:
        if feat == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, feat, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(feat), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = feat
    return layers

concat_list = [[8, 16, 32, 64], [4, 8, 16, 32], [8, 16], [4, 8], [], []]

class ConcatLayer(nn.Module):

    def __init__(self, param_list, scale_param, deconv=True):
        super(ConcatLayer, self).__init__()
        list_len = len(param_list)
        layers = []
        self.deconv = deconv

        for i in range(list_len):
            layers += [nn.ConvTranspose2d(1, 1, kernel_size=param_list[i],
                                          stride=param_list[i]//2, padding=param_list[i]//4)]

        self.upsample = nn.ModuleList(layers)
        self.conv = nn.Conv2d(list_len+1, 1, kernel_size=1, stride=1)
        if deconv:
           self.convTranspose = nn.ConvTranspose2d(1, 1, kernel_size=scale_param*2,
                                                stride=scale_param, padding=scale_param//2)
        else:
            self.convTranspose = None

    def forward(self, x, xlist):
        elem_list = [x]
        for i, elem in enumerate(xlist):
            elem_list.append(self.upsample[i](elem))
        if self.deconv:
            output = self.convTranspose(self.conv(torch.cat(elem_list, dim=1)))
        else:
            output = self.conv(torch.cat(elem_list, dim=1))
        return output


def concat_layers(concatList):
    layers = []
    scale = 1
    for i,group in enumerate(concatList):
        layers += [ConcatLayer(group, scale, i!=0)]
        scale *= 2
    return layers


def side_output():
    layers = [
        nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
        ),
        nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
        ),
        nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
        ),
        nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=5, stride=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
        ),
        nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        )
    ]
    return layers

class FusionLayer(nn.Module):

    def __init__(self, num=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(num))
        self.num = num

    def _average_weights(self):
        nn.init.constant_(self.weights, 1/self.num)

    def forward(self, x):
        self._average_weights()
        for i in range(self.num):
            output = self.weights[i] * x[i] if i == 0 else output + self.weights[i] * x[i]

        return output



class VSDNet(nn.Module):
    def _slow_forward(self):
        super(VSDNet, self).__init__()
        self.vggNet = nn.ModuleList(vgg16(vgg16_layers))
        self.side = nn.ModuleList(side_output())
        self.concat = nn.ModuleList(concat_layers(concat_list))
        self.fusion = FusionLayer()
        self.pooling = nn.AvgPool2d(3, 1, 1)
        self.extract = extract = [3, 8, 15, 22, 29]  # extract the map before every max_pooling operation
        self.aggregate = [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]

    def forward(self, input):
        x = input
        sideFeat = []
        count = 0

        for i in range(len(self.vggNet)):
            x = self.vggNet[i](x)
            if i in self.extract:
                sideFeat.append(self.side[count](x))
                count += 1
        sideFeat.append(self.side[count](self.pooling(x)))  # max_pooling --> average_pooling

        sideOutput = []
        for i in range(len(sideFeat)):
            group = [sideFeat[j] for j in self.aggregate[i]]
            sideOutput += [self.concat[i](sideFeat[i], group)]

        sideOutput.append(self.fusion(sideOutput))

        output = torch.Tensor(sideOutput)
        output = nn.functional.sigmoid(output)

        return output


















