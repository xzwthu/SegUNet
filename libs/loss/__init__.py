import logging
import torch.nn as nn
from .loss import FocalLoss,DiceLoss,DistanceLoss
key2opt = {
    "focal": FocalLoss,
    "crossentropy": nn.CrossEntropyLoss,
    'dice':DiceLoss
}


def get_lossfunction(cfg):
    if cfg["solver"]["loss_function"] is None:
        print("Using CrossEntropy")
        return nn.CrossEntropyLoss

    else:
        loss_name = cfg["solver"]["loss_function"]["loss_type"]
        if loss_name not in key2opt:
            raise NotImplementedError("Loss type {} not implemented".format(loss_name))

        print("Using {} optimizer".format(loss_name))
        return key2opt[loss_name]