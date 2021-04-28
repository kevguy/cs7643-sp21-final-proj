import torch
import torch.nn as nn

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        # Referenced from:
        # Very Deep Convolutional Networks for Large-Scale Image Recognition
        # https://arxiv.org/abs/1409.1556
        # https://arxiv.org/pdf/1409.1556.pdf
        # https://github.com/kuangliu/pytorch-cifar
        # conv<receptive field size>-<number of channels>
        # receptive field size = kernel_size
        # padding is fixed to 1
        # all max pooling layers use stride size 2
        self.cnn_layers = Sequential(
            # 1. conv3-64
            Conv2d(3, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            # 2. conv3-64
            Conv2d(64, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            # 3. MaxPool
            MaxPool2d(kernel_size=2, stride=2),
            # 4. conv3-128
            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            # 5. conv3-128
            Conv2d(128, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            # 6. MaxPool
            MaxPool2d(kernel_size=2, stride=2),
            # 7. conv3-256
            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            # 8. conv3-256
            Conv2d(256, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            # 9. conv1-256
            Conv2d(256, 256, kernel_size=1, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            # 10. MaxPool
            MaxPool2d(kernel_size=2, stride=2),
            # 11. conv3-152
            Conv2d(256, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            # 12. conv3-152
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            # 13. conv1-152
            Conv2d(512, 512, kernel_size=1, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            # 14. MaxPool
            MaxPool2d(kernel_size=2, stride=2),
            # 15. conv3-152
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            # 16. conv3-152
            Conv2d(512, 512, kernel_size=3, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            # 17. conv1-152
            Conv2d(512, 512, kernel_size=1, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            # # 18. MaxPool
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(2048, 4096),
            Dropout(p=0.7),
            Linear(4096, 1000),
            Dropout(p=0.3),
            Linear(1000, 10),
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
        # print(outs.size())
        outs = self.linear_layers(outs)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
