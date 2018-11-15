import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

nclasses = 20 

def createModel():
    model = torchvision.models.inception_v3(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(2048,nclasses)
    model.aux_logits = False
    return model


def InceptionInput2OutputKernelsize(x):
    """
    x=75 => y=1
    x=299 => y=8
    """
    y = (((((x-1)/2 -2 -1)/2 -2 -1)/2 -1)/2 -1)/2
    return y