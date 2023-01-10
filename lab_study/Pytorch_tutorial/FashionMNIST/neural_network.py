import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

class NeuralNetwork11(nn.Module):
    def __init__(self):
        super(NeuralNetwork11, self).__init__()
        self.flatten = nn.Flatten()
        self.info = "28*28 -> 512 -> 10"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512), # hidden
            nn.ReLU(),
            nn.Linear(512, 10), # output
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



class NeuralNetwork12(nn.Module):
    def __init__(self):
        super(NeuralNetwork12, self).__init__()
        self.flatten = nn.Flatten()
        self.info = "28*28 -> 1024 -> 10"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1024),  # hidden
            nn.ReLU(),
            nn.Linear(1024, 10),  # output
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork13(nn.Module):
    def __init__(self):
        super(NeuralNetwork13, self).__init__()
        self.flatten = nn.Flatten()
        self.info = "28*28 -> 360 -> 10"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 360),  # hidden
            nn.ReLU(),
            nn.Linear(360, 10),  # output
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork21(nn.Module):
    def __init__(self):
        super(NeuralNetwork21, self).__init__()
        self.flatten = nn.Flatten()
        self.info = "28*28 -> 512 -> 512 -> 10"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # hidden
            nn.ReLU(),
            nn.Linear(512, 512),  # output
            nn.ReLU(),
            nn.Linear(512, 10),  # output
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork22(nn.Module):
    def __init__(self):
        super(NeuralNetwork22, self).__init__()
        self.flatten = nn.Flatten()
        self.info = "28*28 -> 1024 -> 512 -> 10"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1024),  # hidden
            nn.ReLU(),
            nn.Linear(1024, 512),  # output
            nn.ReLU(),
            nn.Linear(512, 10),  # output
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork23(nn.Module):
    def __init__(self):
        super(NeuralNetwork23, self).__init__()
        self.flatten = nn.Flatten()
        self.info = "28*28 -> 512 -> 360 -> 10"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # hidden
            nn.ReLU(),
            nn.Linear(512, 360),  # output
            nn.ReLU(),
            nn.Linear(360, 10),  # output
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork31(nn.Module):
    def __init__(self):
        super(NeuralNetwork31, self).__init__()
        self.flatten = nn.Flatten()
        self.info = "28*28 -> 1024 -> 512 -> 360 -> 10"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1024),  # hidden
            nn.ReLU(),
            nn.Linear(1024, 512),  # output
            nn.ReLU(),
            nn.Linear(512, 360),  # output
            nn.ReLU(),
            nn.Linear(360, 10),  # output
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork32(nn.Module):
    def __init__(self):
        super(NeuralNetwork32, self).__init__()
        self.flatten = nn.Flatten()
        self.info = "28*28 -> 512 -> 360 -> 180 -> 10"
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # hidden
            nn.ReLU(),
            nn.Linear(512, 360),  # output
            nn.ReLU(),
            nn.Linear(360, 180),  # output
            nn.ReLU(),
            nn.Linear(180, 10),  # output
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits