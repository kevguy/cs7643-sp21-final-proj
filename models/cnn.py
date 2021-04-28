import torch.nn as nn

from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.cnn_layers = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(5408, 10)
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.cnn_layers(x)
        outs = outs.view(outs.size(0), -1)
        outs = self.linear_layers(outs)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs