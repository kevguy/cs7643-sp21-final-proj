import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.linear_layers = Sequential(
            Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            Linear(hidden_size, num_classes)
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out = self.linear_layers(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out