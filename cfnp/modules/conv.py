import torch.nn as nn
from argparse import ArgumentParser

def conv3x1(in_planes, out_planes, stride=1, is_transpose=False):
    """3x1 convolution with padding"""
    if is_transpose:
            return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=(3,1), stride=stride,
                     padding=(1,0), bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,1), stride=stride,
                     padding=(1,0), bias=False)

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, is_transpose=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride, is_transpose=is_transpose)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes, is_transpose=is_transpose)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, is_transpose=False):
        super(Bottleneck, self).__init__()
        if is_transpose:
            self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=(3,1), stride=stride, padding=(1,0), bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3,1), stride=stride, padding=(1,0), bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResCompressionNet(nn.Module):
    def __init__(self, block, layers, r, n_compressed):
        super(ResCompressionNet, self).__init__()
        # r means U_truncated shape[1]
        self.output_dim = (n_compressed,r)
        if r > 512:
            base = 64
        else:
            base = r
        self.inplanes = 2*base
        # self.conv1 = nn.Conv2d(r, 2*r, kernel_size=(7,3), stride=(1,1), padding=(3,1),
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(2*r)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1,1), padding=1)
        # self.layer1 = self._make_layer(block, 2*r, layers[0])
        # self.layer2 = self._make_layer(block, 4*r, layers[1], stride=(1,1))
    
        # self.layer3 = self._make_layer(block, 2*r, layers[2], stride=(1,1))
        # self.layer4 = self._make_layer(block, r, layers[3], stride=(1,1))
        # #TODO: 反卷积？
        self.conv1 = nn.Conv2d(r, 2*base, kernel_size=(7,3), stride=(2,1), padding=(3,1),
                        bias=False)
        self.bn1 = nn.BatchNorm2d(2*base)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2,1), padding=1)
        self.layer1 = self._make_layer(block, 2*base, layers[0])
        self.layer2 = self._make_layer(block, 4*base, layers[1], stride=(1,1))
        self.layer3 = self._make_transpose_layer(block, 2*base, layers[2], stride=(2,1))
        self.layer4 = self._make_transpose_layer(block, base, layers[3], stride=(2,1))
        #TODO: 反卷积？
        self.avgpool1 = nn.AdaptiveAvgPool2d((n_compressed, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((r,1))
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _make_transpose_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: #or planes!=self.output_dim[1]:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, is_transpose=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_transpose=True))
 
        return nn.Sequential(*layers)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: #or planes!=self.output_dim[1]:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        # print('conv1:',x.size())
        x = self.bn1(x)
        # print('bn1:',x.size())
        x = self.relu(x)
        # print('relu:',x.size())
        x = self.maxpool(x)
        # print('maxpool:',x.size())
        x = self.layer1(x)
        # print('layer1:',x.size())
        x = self.layer2(x)
        # print('layer2:',x.size())
        x = self.layer3(x)
        # print('layer3:',x.size())
        x = self.layer4(x)
        # print('layer4:',x.size())
        x = self.avgpool1(x)
        # print('avgpool1:',x.size())
        x = self.avgpool2(x.view(1,self.output_dim[0],-1,1))
        # print('avgpool2:',x.size())
        x = x.view(self.output_dim[0], self.output_dim[1])
        # print('view::',x.size())
 
        return x
    @staticmethod
    def add_specific_args(parent_parser: ArgumentParser):
        return parent_parser

def resnet18(r, n_compressed, **kwargs):
    model = ResCompressionNet(BasicBlock, [2,2,2,2], r, n_compressed, **kwargs)
    return model

def resnet34(r, n_compressed, **kwargs):
    model = ResCompressionNet(BasicBlock,[3,4,6,3], r, n_compressed, **kwargs)
    return model

def resnet50(r, n_compressed, **kwargs):
    model = ResCompressionNet(Bottleneck, [3,4,6,3], r, n_compressed, **kwargs)
    return model
