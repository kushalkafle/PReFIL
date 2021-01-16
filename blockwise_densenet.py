from collections import OrderedDict

from torch import nn
from torchvision.models.densenet import _DenseBlock, _Transition


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 6, 6),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()
        self.block_config = block_config
        # First convolution
        self.first_conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.denseblock = []
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.denseblock.append(nn.Sequential(OrderedDict([
                (f'dblock{i}', block),
            ])))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.denseblock[i].add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_features)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.denseblock = nn.ModuleList(self.denseblock)

    def forward(self, x):
        first_conv_feat = self.first_conv(x)
        denseblock_feat = [self.denseblock[0](first_conv_feat)]

        for i in range(len(self.block_config) - 1):
            denseblock_feat.append(self.denseblock[i + 1](denseblock_feat[i]))
        final_feat = self.final_bn(denseblock_feat[-1])
        return first_conv_feat, denseblock_feat, final_feat
