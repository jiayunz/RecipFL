'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.classifier(out)
        return out


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes, input_channel=1):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        if input_channel == 3:
            self.fc = nn.Linear(400, 120)
        else:
            self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.classifier = nn.Linear(84, num_classes)

    def forward(self, x, return_emb=False):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        if return_emb:
            return out, self.classifier(out)
        else:
            return self.classifier(out)

    def forward_shallow(self, x, return_emb=False):
        out = self.layer1(x)
        out = self.adapter(out)
        out = out.reshape(out.size(0), -1)
        out = self.adapter_linear(out)
        if return_emb:
            return out, self.layer1_classifier(out)
        else:
            return self.layer1_classifier(out)

class LeNet5_dwscaled(nn.Module):
    # conv layer1 will turn to a 1 - 1 channel conv, which seems odd. you may specify it to a 1 - 2 channel conv if needed.
    def __init__(self, num_classes):
        super(LeNet5_dwscaled, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(48, 84)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(84, num_classes)

    def forward(self, x, return_emb=False):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        if return_emb:
            return out, self.classifier(out)
        else:
            return self.classifier(out)

    def forward_shallow(self, x, return_emb=False):
        out = self.layer1(x)
        out = self.adapter(out)
        out = out.reshape(out.size(0), -1)
        out = self.adapter_linear(out)
        if return_emb:
            return out, self.layer1_classifier(out)
        else:
            return self.layer1_classifier(out)

def test():
    from torchsummary import summary
    #net = LeNet1(num_classes=10)
    #summary(net, (1, 28, 28))
    net = LeNet5(num_classes=10).cuda()
    summary(net, (1, 28, 28))
    #net = LeNet5_dwscaled(num_classes=10).cuda()
    #summary(net, (1, 28, 28))

#test()
