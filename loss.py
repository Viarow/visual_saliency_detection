import torch
from torch import nn
import torch.nn.functional as F

class finalLoss(nn.Module):
    def __init__(self, weights=[1.0]*7):
        super(finalLoss, self).__init__()
        self.weights = weights

    def forward(self, pred, label):
        loss = self.weights[0] * F.binary_cross_entropy(pred[0], label)
        for i, x in enumerate(pred[1:]):
            loss += self.weights[i] * F.binary_cross_entropy(x, label)
        return loss

def mean_abs_error(pred, target):
    return torch.abs(target-pred).mean()