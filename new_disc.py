from torch import nn
import torch.nn.functional as F
import torch


class IDiscriminator(nn.Module):
    def __init__(self):
        super(IDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3,
                               padding=1)  # nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True)#, nn.LeakyReLU(0.2,
        # True))
        self.conv2 = nn.Conv2d(64, 64, 3,
                               padding=1)  # nn.Conv2d(64, 64, (4, 4), (2, 2), (1, 1), bias=False)#, nn.BatchNorm2d(
        # 64), nn.LeakyReLU(0.2, True))
        # x2

        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3,
                               padding=1)  # nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False)#,nn.BatchNorm2d(
        # 128), nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Conv2d(128, 128, 3,
                               padding=1)  # nn.Conv2d(128, 128, (4, 4), (2, 2), (1, 1), bias=False)#,
        # nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True))
        # x4
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3,
                               padding=1)  # nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False)#,
        # nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))
        self.conv6 = nn.Conv2d(256, 256, 3,
                               padding=1)  # nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False)#,
        # nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))
        # x6
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(256, 512, 3,
                               padding=1)  # nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False)#nn.Sequential( ,
        # nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        self.conv8 = nn.Conv2d(512, 512, 3,
                               padding=1)  # nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)#,
        # nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        # x8
        self.bn8 = nn.BatchNorm2d(512)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.conv9 = nn.Conv2d(512, 512, 3,
                               padding=1)  # nn.Conv2d(512,512,3,padding=1)#nn.Conv2d(512, 512, (3, 3), (1, 1), (1,
        # 1), bias=False)#, nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        self.conv10 = nn.Conv2d(512, 512, 3,
                                padding=1)  # nn.Conv2d(512,512,3,padding=1)#nn.Conv2d(512, 512, (4, 4), (2, 2), (1,
        # 1), bias=False)#, nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        # x10
        self.pool10 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(499712, 512)  # 64*(64*64) + 128*(32*32) + 256*(16*16) + 512*(8*8) + 512*(4*4)
        self.fc2 = nn.Linear(512, 1)
        self.act = nn.LeakyReLU(0.2, True)
        self.fc3 = nn.Linear(1, 1)

    def flatten(self, x):  # takes NxCxHxW input and outputs NxHWC
        return x.view(x.shape[0], -1)

    def forward(self, input):
        x2 = self.pool2(F.relu(self.bn2(self.conv2(F.relu(self.conv1(input))))))

        x4 = self.pool4(F.relu(self.bn4(self.conv4(F.relu(self.bn4(self.conv3(x2)))))))  # self.conv4(self.conv3(x2))

        x6 = self.pool6(F.relu(self.bn6(self.conv6(F.relu(self.bn6(self.conv5(x4)))))))  # self.conv6(self.conv5(x4))

        x8 = self.pool8(F.relu(self.bn8(self.conv8(F.relu(self.bn8(self.conv7(x6)))))))  # self.conv8(self.conv7(x6))

        x10 = self.pool10(
            F.relu(self.bn8(self.conv10(F.relu(self.bn8(self.conv9(x8)))))))  # self.conv10(self.conv9(x8))

        # flatten and concatenate
        features = torch.cat(
            (self.flatten(x2), self.flatten(x4), self.flatten(x6), self.flatten(x8), self.flatten(x10)), 1)

        return self.fc3(0.01 * self.fc2(self.act(self.fc1(features))))
