import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(torch.Tensor(np.array(alpha)))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        targets = targets.long()
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(inputs.shape).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets[:,None,:,:,:]
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[targets]

        probs = (P*class_mask).sum(1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
loss_l2 = nn.MSELoss()
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Arguments:
            input: (B, C, H, W)

        Return:
            loss
        """
        # (B, _, H, W) = input.shape
        targets = targets.long()
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(inputs.shape).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets[:,None,:,:,:]
        class_mask.scatter_(1, ids.data, 1.)
        true_positive = (P*class_mask).sum((2,3,4))
        dice_score = 2*true_positive/(P.sum((2,3,4))+class_mask.sum((2,3,4)))
        return 1-dice_score.mean()

class GradLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(myLoss2, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = 2
    
    def forward(self, input, target):
        """
        Arguments:
            input: (B, C, H, W)

        Return:
            loss
        """
        # (B, _, H, W) = input.shape
        preds = input
        labels = target
        eff1 = torch.pow((2-preds),self.gamma)
        eff2 = torch.pow(1+preds,self.gamma)
        # loss = -(target*torch.log(preds+1e-9)*eff1+(1-target)*torch.log(1-preds+1e-9)*eff2).mean()
        loss = -((target==1).float()*torch.log(preds+1e-9)*eff1+(target==0).float()*torch.log(1-preds+1e-9)*eff2).mean()
        return loss
class DistanceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(myLoss3, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = 2
    
    def forward(self, input, target):
        """
        Arguments:
            input: (B, C, H, W)

        Return:
            loss
        """
        # (B, _, H, W) = input.shape
        preds = input
        labels = target
        zero_sum = (labels==0).sum()
        one_sum = (labels==1).sum()
        # w_zero = one_sum/zero_sum
        w_one = zero_sum/one_sum
        eff1 = torch.pow((1-preds),self.gamma)
        eff2 = torch.pow(preds,self.gamma)
        loss = -(target*torch.log(preds+1e-9)*eff1*w_one+(1-target)*torch.log(1-preds+1e-9)*eff2).mean()
        return loss*2

