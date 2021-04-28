import torch
import torch.nn as nn
import torch.nn.functional as F

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    # calculate - (1 - beta) / (1 - beta^n) for each class, and then normalize
    # Referenced from: https://zhuanlan.zhihu.com/p/142496614
    # print(torch.tensor(cls_num_list))
    # print(beta)
    # print(torch.pow(beta, torch.tensor(cls_num_list).type(torch.DoubleTensor)))
    effective_num = 1.0 - torch.pow(beta, torch.tensor(cls_num_list).type(torch.DoubleTensor))
    weights = (1.0 - beta) / ((effective_num)+1e-8)
    per_cls_weights = F.normalize(weights, p=1, dim=0)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        weights = torch.tensor(self.weight, device=input.device).index_select(0, target)
        weights = weights.unsqueeze(1)

        # loss = focal_loss(input, target, weights, self.gamma)

        # Referenced from: https://blog.csdn.net/nathan_yo/article/details/106482850
        labels_one_hot = F.one_hot(target, 10).float()

        # Sigmoid + Cross Entropy
        # bce_loss = F.binary_cross_entropy_with_logits(input=input, target=labels_one_hot, reduction="none")

        # Softmax + Cross Entropy
        # Changed to this according to suggestions from TA on Piazza
        bce_loss = F.binary_cross_entropy(input=F.softmax(input), target=labels_one_hot, reduction="none")

        modulator = torch.exp(-self.gamma * labels_one_hot * input - self.gamma * torch.log(1 + torch.exp(-1.0 * input)))

        loss = modulator * bce_loss

        weighted_loss = weights * loss
        loss = torch.sum(weighted_loss)

        # Not necessary, use mean instead
        # loss /= torch.sum(labels)
        loss = loss.mean()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
