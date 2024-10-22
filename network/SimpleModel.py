import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class SimpleModel(nn.Module):

    def __init__(self, config):
        super(SimpleModel, self).__init__()

        self.config = config
        self.model = EfficientNet.from_pretrained('efficientnet-b2')
        self.model._fc = nn.Linear(in_features=1408, out_features=self.config['backbone_config']['num_classes'])
        # initial params
        torch.nn.init.normal_(self.model._fc.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == "__main__":
    import yaml
    # 加载配置文件
    with open('training/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    efficientnet = SimpleModel(config)
    output = efficientnet(torch.rand((4,3,256,256))) # 前向传播
    print(output)
