import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, filters, dropout, activation, use_batchnorm, padding):
        super(CNN, self).__init__()
        act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
        pad = 1 if padding == "same" else 0

        self.conv1 = nn.Conv2d(3, filters, kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(filters) if use_batchnorm else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(filters * 16 * 16, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 10)
        self.activation = act_fn

    def forward(self, x):
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
