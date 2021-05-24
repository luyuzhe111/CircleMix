import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.nll_loss = nn.NLLLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        if self.reduce:
            return focal_loss.mean()
        else:
            return focal_loss

        # log_p = F.log_softmax(inputs, dim=-1)
        # ce = self.nll_loss(log_p, targets)
        # all_rows = torch.arange(len(inputs))
        # log_pt = log_p[all_rows, targets]
        #
        # pt = log_pt.exp()
        # focal_term = (1 - pt) ** self.gamma
        # F_loss = focal_term * ce
        # if self.reduce:
        #     return torch.mean(F_loss)
        # else:
        #     return F_loss


class EffectiveSamplesLoss(nn.Module):
    def __init__(self, beta=0.9, num_cls=2, sample_per_cls=None, focal=None, focal_gamma=2, focal_alpha=1):
        super(EffectiveSamplesLoss, self).__init__()
        self.beta = beta
        self.num_cls = num_cls
        self.effective_num = 1.0 - np.power(self.beta, sample_per_cls)
        self.nll_loss = nn.NLLLoss(reduction='none')
        self.focal = focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(self, inputs, targets):
        weights = torch.from_numpy( (1 - self.beta) / np.array(self.effective_num) ).float().cuda()
        weights = weights / torch.sum(weights) * self.num_cls
        log_p = F.log_softmax(inputs, dim=-1)
        CELoss = self.nll_loss(log_p, targets)
        if self.focal:
            all_rows = torch.arange(len(inputs))
            log_pt = log_p[all_rows, targets]
            pt = log_pt.exp()
            focal_term = (1 - pt) ** self.focal_gamma
            FLoss = self.focal_alpha * focal_term * CELoss
            ESLoss = weights[targets] * FLoss
        else:
            ESLoss = weights[targets] * CELoss

        return torch.mean(ESLoss)
