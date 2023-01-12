"""
Neural network implementation
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBatchNorm2d(nn.Module):

    """
    Adaptative batch normalization implementation
    """

    def __init__(self, num_features, momentum=.1, eps=1e-5, affine=True):

        """
        Class constructor

        An adaptative batch normalization layer takes as input a tensor x and 
        outputs a tensor y defined by

        .. math::
        y = a BN(x) + bx

        where BN is a batch normalization layer, and a and b are learnable 
        parameters. 
       
        .. math::
        BN(x) = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + 
          \epsilon}} * \gamma + \beta

        The shape of the input tensor is BxCxWxH where B is the number of 
        images in each batch, C the number of features maps, W the width 
        of the image and H the height of the image, respectively.

        Attributes
        ----------
        num_features: int
           number of features map
        momentum: float
           parameter used by the batch normalizatio layer to compute the 
           statistics
        eps: float
           value added to the denominator for ensuring stability
        affine: boolean 
           when set to True, indicates that the batch normalization 
           layer has learnable affine parameters
        """

        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, momentum, eps, affine)
        tens_a = torch.FloatTensor(1, 1, 1, 1)
        tens_b = torch.FloatTensor(1, 1, 1, 1)
        tens_a[0, 0, 0, 0] = 1
        tens_b[0, 0, 0, 0] = 0
        self.a = nn.Parameter(tens_a)
        self.b = nn.Parameter(tens_b)


    def forward(self, x):

        """
        Forward pass in the adaptative batch normalization layer
        
        .. math::
        y = a BN(x) + bx

        where BN is a batch normalization layer, and a and b are learneable 
        parameters

        Parameters
        ----------
        x: pytorch tensor
           input tensor, with size BxCxWxH

        Returns
        -------
        out: pytorch tensor
           transformed tensor
        """
        return self.a * x + self.b * self.bn(x)


# ------------------------------------
# 1: Dilated convolution module
# ------------------------------------

class BaseBlock(nn.Module):

    """
    Convolution module implementation: 2D dilated Convolution followed 
    by a batch normalization layer and a leaky RELU activation function

    Attributes
    ----------

    conv: instance of nn.Conv2d
       2D convolution module
    conv_abn: instance of AdaptiveBatchNorm2d
       Adaptative batch normalization module
    LReLU: instance of nn.LeakyReLU
       Leaky ReLU non-linearity
    """

    def __init__(self, in_channels, out_channels, s):

        """
        Constructor

        Parameters
        ----------
        in_channels: int
           shape of the input tensor
        out_channels: int
           shape of the output tensor
        s: int
           dilation scale
        """

        super(BaseBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
          padding = 2**(s-1), dilation = 2**(s-1), bias=True)
        self.conv_abn = AdaptiveBatchNorm2d(out_channels)
        self.LReLU = nn.LeakyReLU(0.2, inplace = True)


    def forward(self, x):

        """
        Forward pass in the convolution module

        Parameters
        ----------
        x: pytorch tensor
           input tensor, with size BxCxWxH

        Returns
        -------
        out: pytorch tensor
           transformed tensor
        """

        return self.LReLU(self.conv_abn(self.conv(x)))


# ------------------------------------
# 3: Neural network architecture
# ------------------------------------

class Net(nn.Module):

    """
    Implementation of the neural network architecture
    """

    def __init__(self, d=7, w=24):

        """
        Class constructor

        Parameters
        ----------
        d: int
           number of blocks in the architecture
        """

        super(Net, self).__init__()
        self.first_layer = BaseBlock(3, w, 1)
        
        self.intermediate_layers = nn.ModuleList([BaseBlock(w, w, s) 
          for s in range(2, d)])
        self.final_layer = nn.Conv2d(w, 2, kernel_size=1, bias=True)

      


    def forward(self, x):

        """
        Forward pass in the convolutional network

        Parameters
        ----------
        x: pytorch tensor
           input tensor, with size BxCxWxH

        Returns
        -------
        out: pytorch tensor
           transformed tensor
        """

        x = self.first_layer(x)
        for l in self.intermediate_layers:
            x = l(x)
        x = self.final_layer(x)
        return x


    def num_flat_features(self, x):

        """
        Size of the flattened tensor

        Parameters
        ----------

        x: pytorch tensor
           Input tensor with size BxCxWxH (B: batch size, 
            C: number of channels, W: image width, H: image height)

        Returns
        -------

        out: int
           CxWxH
        """

        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



