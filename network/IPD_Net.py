import torch
import torch.nn as nn
try:
    from network.util.resnet import resnet50
    from network.util.NL import NLBlockND
    from network.util.SRM import SRMConv2d_simple
except:
    from util.resnet import resnet50
    from util.NL import NLBlockND
    from util.SRM import SRMConv2d_simple


class IPD_Net(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.res = resnet50()

        self.PCL = NLBlockND(in_channels=2048, dimension=2, mode="dot")

        self.SRM_k = SRMConv2d_simple(inc=3, learnable=False)

        self.a_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 2)

    def forward(self, x):

        # learn from patchcraft,SRM first
        x = self.SRM_k(x)

        # basic opration,conv some times
        x = self.res.conv1(x)
        x = self.res.bn1(x)
        x = self.res.relu(x)
        x = self.res.maxpool(x)

        # resnet50 as backbone
        x = self.res.layer1(x)
        x = self.res.layer2(x)
        x = self.res.layer3(x)
        x = self.res.layer4(x)

        # cal a PCL M
        x = self.PCL(x)

        # adaptive pool
        x = self.a_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    # 生成一个随机的 (N, C, H, W) 张量，假设 N=8, C=3, H=224, W=224
    N, C, H, W = 2, 3, 512, 512
    input_tensor = torch.randn(N, C, H, W)

    model = IPD_Net(None)

    # 通过网络传递输入张量
    output = model(input_tensor)

    # 打印输出张量的形状
    print("Output shape:", output.shape)
