'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channel=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_emb=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if return_emb:
            return out, self.classifier(out)
        else:
            return self.classifier(out)

    def forward_shallow(self, x, return_emb=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.adapter(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if return_emb:
            return out, self.layer1_classifier(out)
        else:
            return self.layer1_classifier(out)





class ResNet_wscaled(nn.Module):
    def __init__(self, block, num_blocks, input_channel=3, num_classes=10, scale_ratio=0.25):
        super(ResNet_wscaled, self).__init__()
        self.in_planes = int(64 * scale_ratio)

        self.conv1 = nn.Conv2d(input_channel, int(64 * scale_ratio), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64 * scale_ratio))
        self.layer1 = self._make_layer(block, int(64 * scale_ratio), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * scale_ratio), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * scale_ratio), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * scale_ratio), num_blocks[3], stride=2)
        self.classifier = nn.Linear( int(512*block.expansion * scale_ratio), num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_emb=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if return_emb:
            return out, self.classifier(out)
        else:
            return self.classifier(out)


class ResNet_1block(nn.Module):
    def __init__(self, num_blocks, input_channel=3, num_classes=10):
        super(ResNet_1block, self).__init__()
        block = BasicBlock
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.adapter = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, stride=8, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer1_classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_emb=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.adapter(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if return_emb:
            return out, self.layer1_classifier(out)
        else:
            return self.layer1_classifier(out)


def ResNet18(input_channel, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_channel, num_classes)

def ResNet18_wscaled(input_channel, num_classes, scale_ratio=0.2):
    return ResNet_wscaled(BasicBlock, [2, 2, 2, 2], input_channel, num_classes, scale_ratio)

def ResNet34(input_channel, num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], input_channel, num_classes)


def ResNet50(input_channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], input_channel, num_classes)


def ResNet101(input_channel, num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], input_channel, num_classes)


def ResNet152(input_channel, num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], input_channel, num_classes)


def test():
    from torchsummary import summary
    net = ResNet_1block(num_blocks=[2], input_channel=3, num_classes=10).cuda()
    summary(net, (3, 32, 32))
    net = ResNet18(3, 10).cuda()
    summary(net, (3, 32, 32))

