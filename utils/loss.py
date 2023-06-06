import math
from torch import nn
import torch 
class Divine2(nn.Module):
    def __init__(self, num_di = 5):
        super().__init__()
        self.divine = num_di
    def forward(self, x):
        return x / self.divine

class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        '''
        :param pred: Bx30(Landmark)
        :param target: Bx30(Landmark)
        :return:
        '''
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
 
    def forward(self, pred, target):
        '''
        :param pred: Bx30(Landmark)
        :param target: Bx30(Landmark)
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class L2Loss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(L2Loss, self).__init__()
        self.epsilon = epsilon
    def forward(self, pred, target):
        '''
        :param pred: Bx30(Landmark)
        :param target: Bx30(Landmark)
        :return:
        '''
        return torch.mean(((target - pred) ** 2) + self.epsilon, dim=1).mean() 

class L1Loss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(L1Loss, self).__init__()
        self.epsilon = epsilon
    def forward(self, pred, target):
        '''
        :param pred: Bx30(Landmark)
        :param target: Bx30(Landmark)
        :return:
        '''
        return torch.mean(torch.abs_(target - pred) + self.epsilon, dim=1).mean()
    
