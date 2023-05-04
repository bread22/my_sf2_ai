import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        fc_input_shape = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(fc_input_shape, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_conv_output(self, shape):
        x = torch.zeros(shape)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(x.reshape(1, -1).size()[1])
