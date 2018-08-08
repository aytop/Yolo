from torch import nn
from torch.nn import functional as f
from torchsummary import summary


class YOLONet(nn.Module):
    def __init__(self):
        super(YOLONet, self).__init__()
        self.c1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.c21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.c22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.c31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.c32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.c42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.c43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d((2, 2))
        self.mp2 = nn.MaxPool2d((1, 2))
        self.c51 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.c52 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.c53 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(256, 15, kernel_size=3, padding=1)
        summary(self, (3, 72, 144))

    def forward(self, x):
        x = f.relu(self.c1(x))
        x = self.mp1(x)

        x = f.relu(self.c21(x))
        x = f.relu(self.c22(x))
        x = self.mp1(x)

        x = f.relu(self.c31(x))
        x = f.relu(self.c32(x))
        x = f.relu(self.c33(x))
        x = self.mp1(x)

        x = f.relu(self.c41(x))
        x = f.relu(self.c42(x))
        x = f.relu(self.c43(x))
        x = self.mp2(x)

        x = f.relu(self.c51(x))
        x = f.relu(self.c52(x))
        x = f.relu(self.c53(x))

        x = self.c6(x)

        return x
