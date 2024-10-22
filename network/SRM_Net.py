import torch
import torch.nn as nn
from network.util.SRM import SRMConv2d_simple
import timm

class SRM_Net(nn.Module):

    def __init__(self, name, pretrained):
        super().__init__()

        self.SRM_k = SRMConv2d_simple(inc=3, learnable=False)

        self.model = timm.create_model(name, pretrained=pretrained, num_classes=2)

    def forward(self, x):

        # learn from patchcraft,SRM first
        x = self.SRM_k(x)

        x = self.model(x)

        return x
