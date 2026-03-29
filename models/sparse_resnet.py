import torch.nn as nn
import torch.nn.functional as F

class SparseResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        mask = (x.abs().sum(dim=1, keepdim=True) > 0).float()
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)

        return out * mask


class SparseResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = SparseResidualBlock(8,16)
        self.b2 = SparseResidualBlock(16,32)
        self.b3 = SparseResidualBlock(32,64)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        x = self.b1(x)
        x = F.max_pool2d(x,2)

        x = self.b2(x)
        x = F.max_pool2d(x,2)

        x = self.b3(x)
        x = self.pool(x).view(x.size(0),-1)

        return self.fc(x)