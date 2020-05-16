from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_num, num, stride):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_num, num, kernel_size=1, bias=False),
            nn.BatchNorm2d(num),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num, num, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(num),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num, num * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(num * 4)
        )

        self.conv4 = nn.Sequential()
        if in_num != num * 4 or stride != 1:
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_num, num * 4, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(num * 4)
            )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        residual = self.conv4(x)
        return nn.ReLU(inplace=True)(residual + y)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_x = self._make_layer(16, 3, 1)
        self.conv3_x = self._make_layer(32, 4, 2)
        self.conv4_x = self._make_layer(64, 6, 2)
        self.conv5_x = self._make_layer(128, 3, 2)
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, 58)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * 4

        for i in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

# class cnn(nn.Module):
#     def __init__(self):
#         super(cnn, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3,
#                 out_channels=16,
#                 kernel_size=5,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),  # => 16*256*256
#             nn.BatchNorm2d(16),
#             nn.MaxPool2d(kernel_size=2),  # => 16*128*128
#             nn.Dropout2d(0.2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 5, 1, 2),
#             nn.ReLU(),  # => 32*128*128
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=2),  # => 32*64*64
#             nn.Dropout2d(0.2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, 5, 1, 2),
#             nn.ReLU(),  # => 64*64*64
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=2),  # => 64*32*32
#             nn.Dropout2d(0.2)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 128, 5, 1, 2),
#             nn.ReLU(),  # => 128*32*32
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(kernel_size=2),  # => 128*16*16
#             nn.Dropout2d(0.5)
#         )
#         self.out = nn.Linear(128 * 16 * 16, 58)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = x.reshape(x.size(0), -1)
#         output = self.out(x)
#         return output
