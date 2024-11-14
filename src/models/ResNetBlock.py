import collections

import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    '''
        ResNet50, ResNet101, ResNet152, ... generic block implementation
    '''
    
    def __init__(
            self,
            index                    : int, 
            in_channels              : int,
            intermediate_channels    : int,
            expansion                : int,
            kernel_sizes             : list[int],
            mid_module_stride        : int,
            paddings                 : list[int],
            identity_channel_scaling : None | nn.Module):
        
        super(ResNetBlock, self).__init__()

        # Ordered Dictionary for storing the modules with distinctive names
        modules_dict : collections.OrderedDict = collections.OrderedDict()
        
        # These blocks always have three modules
        modules_dict[f"conv{index}_1"] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=paddings[0])
        modules_dict[f"batchnorm{index}_1"] = nn.BatchNorm2d(num_features=intermediate_channels)

        modules_dict[f"conv{index}_2"] = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_sizes[1],
            stride=mid_module_stride,
            padding=paddings[1])
        modules_dict[f"batchnorm{index}_2"] = nn.BatchNorm2d(num_features=intermediate_channels)

        out_channels : int = intermediate_channels * expansion

        modules_dict[f"conv{index}_3"] = nn.Conv2d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=paddings[2])
        modules_dict[f"batchnorm{index}_3"] = nn.BatchNorm2d(num_features=out_channels)

        # Store the identity downsample module object as an attribute for later use
        self.identity_channel_scaling : nn.Module = identity_channel_scaling

        # Create the block a module object
        self.block : nn.Sequential = nn.Sequential(modules_dict)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Store the original input in order to add it at the end of the block
        identity : torch.Tensor = x

        # Pass the input through the block
        x = self.block(x)

        # Check if the input was downsampled. In that case, downsample the identity
        if self.identity_channel_scaling != None:
            identity = self.identity_channel_scaling(identity)

        # Add the forwarded input and the original input (downsampled if needed)
        x += identity

        # Pass through the ReLU activation function
        x = nn.ReLU(x)

        return x