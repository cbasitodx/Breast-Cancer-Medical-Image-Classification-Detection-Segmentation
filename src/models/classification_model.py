import torch
import torch.nn as nn
import collections

from models.ResNetBlock import ResNetBlock

class ClassificationModel(nn.Module):
    '''
        ResNet50, ResNet101, ResNet152, ... implementation
    '''
        
    def __init__(self, input_image_channels : int, num_classes : int, n_reps_per_block : list[int]):
        super(ClassificationModel, self).__init__()

        # Set the number of input channels at the beginning of the deep layers
        self.in_channels : int = 64

        # Initial layer (it's exactly the same for every ResNet)
        self.conv1_layer : nn.Sequential = nn.Sequential(collections.OrderedDict([
            (
                'conv1_1',
                nn.Conv2d(in_channels=input_image_channels,out_channels=self.in_channels,kernel_size=7,stride=2,padding=3, bias=False)
            ),
            (
                'batchnorm1_1',
                nn.BatchNorm2d(num_features=self.in_channels)
            ),
            (
                'ReLU1',
                nn.ReLU()
            ),
            (
                'maxpool1',
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        ]))

        # List containing the repetitions of each block in the net. For instance, in ResNet50 this would be [3,4,6,3]
        self.n_reps_per_block : list[int] = n_reps_per_block

        # ResNet deep layers parameters
        kernel_sizes : list[int] = [1,3,1]
        paddings : list[int] = [0,1,0]
        expansion : int = 4

        # ResNet deep layers
        self.conv2_layer : nn.Sequential = self.__make_layer(
            index=2, 
            intermediate_channels=64, 
            expansion=expansion, 
            kernel_sizes=kernel_sizes, 
            mid_module_stride=1, 
            paddings=paddings, 
            n_reps=n_reps_per_block[0])
        
        self.conv3_layer : nn.Sequential = self.__make_layer(
            index=3, 
            intermediate_channels=128, 
            expansion=expansion, 
            kernel_sizes=kernel_sizes, 
            mid_module_stride=2, 
            paddings=paddings, 
            n_reps=n_reps_per_block[1])
        
        self.conv4_layer : nn.Sequential = self.__make_layer(
            index=4, 
            intermediate_channels=256, 
            expansion=expansion, 
            kernel_sizes=kernel_sizes, 
            mid_module_stride=2, 
            paddings=paddings, 
            n_reps=n_reps_per_block[2])
        
        self.conv5_layer : nn.Sequential = self.__make_layer(
            index=5, 
            intermediate_channels=512, 
            expansion=expansion, 
            kernel_sizes=kernel_sizes, 
            mid_module_stride=2, 
            paddings=paddings, 
            n_reps=n_reps_per_block[3])
        
        # Output layers        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)
        
    def __make_layer(
            self,
            index                 : int, 
            intermediate_channels : int,
            expansion             : int,
            kernel_sizes          : list[int],
            mid_module_stride     : int,
            paddings              : list[int],
            n_reps                : int) -> nn.Sequential:
        
        # List of blocks that comprises the layer
        blocks : list[tuple[str, ResNetBlock]] = []

        # Identity channel scaling module
        identity_channel_scaling = None

        # We perform channel scaling between blocks and between layers.
        # Its only going to be performed when the block has a stride different from 1 and when the number of input channels doesnt match the number of output channels (intermediate_channels * expansion)
        if mid_module_stride != 1 or self.in_channels != intermediate_channels * expansion:

            out_channels_block : int = intermediate_channels * expansion
            identity_channel_scaling : nn.Module | None = nn.Sequential(collections.OrderedDict([
                (
                    f"conv_scaling{index}",
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=out_channels_block,
                        kernel_size=1,
                        stride=mid_module_stride,
                        padding=0,
                        bias=False
                    )
                ),
                (
                    f"batchnorm_scaling{index}",
                    nn.BatchNorm2d(out_channels_block)
                )
    
            ]))
        
        # We add the first block. This block connects to the previous layer (list of blocks).
        # For instance, in the first layer of ResNet50, this block goes from 64ch -> 64ch -> 256ch
        blocks.append(
            (
                f"block{index}_1",
                ResNetBlock(index, self.in_channels, intermediate_channels, expansion, kernel_sizes, mid_module_stride, paddings, identity_channel_scaling)
            )
        )

        # Now, we update the number of input channels so the output of the appended block can connect adequately with the next block (next block could be on another layer).
        # For instance, in the first layer of ResNet50, the input size is now 64*4 = 256
        self.in_channels = intermediate_channels * expansion

        # Lastly, append the last blocks.
        # For instance, in the first layer of ResNet50, these last blocks go from 256ch -> 64ch -> 256ch
        # (identity_channel_scaling is set to None because its not needed anymore, and stride is set to 1 because we are not doing any more channel_scaling)
        for i in range(1, n_reps):
            blocks.append(
                (
                    f"block{index}_{i+1}",
                    ResNetBlock(index, self.in_channels, intermediate_channels, expansion, kernel_sizes, 1, paddings, None)
                )
            )
         
        # Concatenate the blocks into a layer
        layer : nn.Sequential = nn.Sequential(collections.OrderedDict(blocks))
        return layer

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Input layer
        x = self.conv1_layer(x)
        
        # Deep residual layers
        x = self.conv2_layer(x)
        x = self.conv3_layer(x)
        x = self.conv4_layer(x)
        x = self.conv5_layer(x)

        # Output layer
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x