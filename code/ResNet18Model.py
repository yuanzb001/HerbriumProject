import torch.nn as nn
from torch.nn import functional as F

class ResNet18BasicBlock(nn.Module):
    def __init__(self, in_channel, outs, kernerl_size, stride, padding):
        super(ResNet18BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.outs = outs
        self.conv1 = nn.Conv2d(self.in_channel, self.outs, kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(self.outs)
        self.conv2 = nn.Conv2d(self.outs, self.outs, kernel_size=kernerl_size[0], stride=stride[1], padding=padding[0])
        self.bn2 = nn.BatchNorm2d(self.outs)
        self.extra = nn.Sequential(
            nn.Conv2d(self.in_channel, self.outs, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(self.outs)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.in_channel != self.outs:
            X_shortcut = self.extra(x)
            out = out + X_shortcut

        return F.relu(out + self.extra)

class ResNet18(nn.Module):
    def __init__(self, class_size):
        super(ResNet18, self).__init__()

        self.preconv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)
        )
        self.Blk1 = ResNet18BasicBlock(64, 64, kernerl_size=[3, 1], stride=[1, 1], padding=[1, 0])
        self.Blk2 = ResNet18BasicBlock(64, 64, kernerl_size=[3, 1], stride=[1, 1], padding=[1, 0])
        self.Blk3 = ResNet18BasicBlock(64, 128, kernerl_size=[3, 1], stride=[2, 1], padding=[1, 0])
        self.Blk4 = ResNet18BasicBlock(128, 128, kernerl_size=[3, 1], stride=[1, 1], padding=[1, 0])
        self.Blk5 = ResNet18BasicBlock(128, 256, kernerl_size=[3, 1], stride=[2, 1], padding=[1, 0])
        self.Blk6 = ResNet18BasicBlock(256, 256, kernerl_size=[3, 1], stride=[1, 1], padding=[1, 0])
        self.Blk7 = ResNet18BasicBlock(256, 512, kernerl_size=[3, 1], stride=[2, 1], padding=[1, 0])
        self.Blk8 = ResNet18BasicBlock(512, 512, kernerl_size=[3, 1], stride=[1, 1], padding=[1, 0])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, class_size)

    def forward(self, x):
        out = self.preconv(x)
        out = self.Blk1(out)
        out = self.Blk2(out)
        out = self.conv1(out) + self.Blk3(out)
        out = self.Blk4(out)
        out = self.conv2(out) + self.Blk5(out)
        out = self.Blk6(out)
        out = self.conv3(out) + self.Blk7(out)
        out = self.Blk8(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out