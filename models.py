import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class RNet(ResNet):
    def __init__(self, block=Bottleneck, pretrained=False, num_classes=102):
        """
        基于resnet152模型的迁移学习
        :param block:
        :param arch:
        :param pretrained:
        :param progress:
        :param num_classes:
        :param kwargs:
        """
        super(RNet, self).__init__(block, layers=[3, 8, 36, 3])

        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls['resnet152'],
                progress=True,
                model_dir='./model/resnet/'
            )
            self.load_state_dict(state_dict)

            for param in self.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(self.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x
