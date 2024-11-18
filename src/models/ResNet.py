# imports
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# Neural net
import torch
import torch.nn as nn # neural network

# Structure of one of the blocks on the table
# output = 4 * input (is going to be expansion * input with expansion = 4)

# Example of block
# conv2_x of 50-layer 
#
# 1 x 1, 64     --> input layer 
# 3 x 3, 64     --> 3 x 3 conv layers. Maintains input_size, and also output size until last layer, which maintains input size but has to do * 4 on the output so it has the same size as the output layer of the block
# 1 x 1, 256    --> output layer. Has a size 4x bigger. This 4 is going to be the variable "expansion"
class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        # identity downsample is going to be done by using a conv layer 
        # we might need to do identitiy downsample if we've changed the input size or the number of channels (example change of dimensions between three layers ago and now)
        super(block, self).__init__()
        self.expansion = 4 # we set expansion to 4 because our blocks will always x4 the input size (see table on paper)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(out_channels) # we normalize the output of each conv layer before inputting it to the next one
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1) #stride is now input to __init__ instead of 1. idk why
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0) # output size will be 4 * input (like in the table of the paper)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU() # we define activation function 
        self.identity_downsample = identity_downsample #"Conv layer that we're going to do to the identity mapping so that it's the same shape later on in the layers" ???????? que

    def forward(self, x):
        identity = x

        # we go trough each layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # we call activation function after each conv layer (+ its normalization)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # After a block we add the identity
        # I think this is in the diagram how some layers take the input of the prev layer AND the input of three layers ago
        if self.identity_downsample is not None:
            # we use the identity downsample if we need to change the shape in some way
            identity = self.identity_downsample(identity)
        
        x += identity # x is now the sum of the output of the last layer and the output of three layers ago
        x = self.relu(x) # We call activation function after last conv later + normalization AND anfter ading the result of three layers ago to it
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        # block -> block class

        # layers -> list of how many times we want to use each passed down block
        #    --> See table -> In each conv layer its [layer] x num because we use that block num times
        #       --> Thats whats defined in the layers list. The first element is for the first block, seccond for the seccond block etc.
        #    --> Example 50 layer (and also 34 layer) architecture = [3,4,6,3] (first and last blocks are used 3 times in their layers, seccond layer uses its block 4 times and third usees it 6 times)

        # image chanells -> num of channels from the input (3 channels for rgb, 1 for black / white, ...) (made so resNet implementation is more general)
        super(ResNet, self).__init__()

        self.in_channels = 64 
        # initial layers. We haven't done any resonant layers yet
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride = 2, padding = 3) # output will always be 64 for conv 1.
        self.bn1 = nn.BatchNorm2d(64) 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1) # I DONT GET WHY IT SETS THE VALUES OF KERNEL AND STRIDE THAT WAY WHAT DOES IT MEAN

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride = 1) # i dont get where it gets the stride from
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride = 2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride = 2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride = 2) # 512 * 4 = 2048 channels at the end of this layer

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # we define output size and it does avg depending on the size we want
        self.fc = nn.Linear(512*4, num_classes) # fully connected layer at the end, mapped to the number of classes we want the model to classify our data into
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x)# we go trough the initial layer (not relational) to change size to 64 independently of original input size

        # the cool layers
        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # output 
        x = self.avgpool(x) # to make it output 1x1
        x = x.reshape(x.shape[0], -1) # reshape it so we can send it to the fully connected layer
        x = self.fc(x)

        return x



    # function to create a resnet layer using our block, how many times we want it to be used, in chanells, out chanells
    def _make_layer(self, block, num_residual_block, out_channels, stride):
        identity_downsample = None
        layers = [] 

        # when are we going to do an identity downsample
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride = stride), # i get what this is supposed to do (with the changing size thing) but not how it works
                                                nn.BatchNorm2d(out_channels*4))
            
        # 64 ->

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride)) 
        # first block is going to change the number of channels (in the case of 50layers, from 64 to 256 (we input 64 but inside of the block its mutiplied by 4 using expansion)
        self.in_channels = out_channels*4 # in channels is the same as the output of the last layer

        for i in range(num_residual_block - 1): # -1 because we already computed one residual block
            layers.append(block(self.in_channels, out_channels)) # These wont change num of chanels, so we input no identity downsample and leave stride as 1 (the default)
            # input value of out_chanels is 64 again. These layers will have an input of 256 and they are going to map it to 256 again going trough intermediate conv layers where it goes 256 -> 64 -> 256
        
        # -> 256

        return nn.Sequential(*layers)


# define the diferent resNets (50, 101 and 152 layer)
# only difference is the num of times the blocks are used on each conv layer (see table on paper) -> layers list argument
def ResNet50(img_channels = 3, num_classes = 1000):
    return ResNet(block, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels = 3, num_classes = 1000):
    return ResNet(block, [3,4,23,3], img_channels, num_classes)

def ResNet152(img_channels = 3, num_classes = 1000):
    return ResNet(block, [3,8,36,3], img_channels, num_classes)



def test():
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet152(img_channels=3, num_classes=1000)
    y = net(torch.randn(BATCH_SIZE, 3, 224, 224)).to(device)
    assert y.size() == torch.Size([BATCH_SIZE, 1000])
    print(y.size())


test()
        
# cosas que no entiendo
# el porque de los paddings, strides y kernel sizes en las definiciones de los bloques
# puse sin querer padding 0 en vez de 1 en la def de blocke en la linea 26 y me soltó un error 
# no entiendo porque queremos esos parametros en especifico