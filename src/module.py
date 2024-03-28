import os

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import transforms

from src.transform import training_transforms


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(L.LightningModule):
    def __init__(self, block, layers, batch_size=2 ** 10, learning_rate=1e-5, device='cpu'):
        super(ResNet, self).__init__()
        self.save_hyperparameters()

        self.inplanes = 64
        self.loss = nn.CrossEntropyLoss()
        self.transform = training_transforms
        self.num_classes = 10
        self.accuracy = Accuracy('multiclass', num_classes=self.num_classes)
        self.set_device = device
        self.in_dim = 1  # cuz it's gray scale!

        self.conv1 = nn.Conv2d(self.in_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def dataset_hp(self):
        return {
            'pin_memory': True,
            'batch_size': self.hparams.batch_size,
            'num_workers': os.cpu_count() - 1,
            'generator': torch.Generator(device=self.set_device),
            'persistent_workers': True,
        }

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        loader = DataLoader(dataset, shuffle=True, **self.dataset_hp)
        return loader

    def val_dataloader(self):
        dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        loader = DataLoader(dataset, **self.dataset_hp)
        return loader

    def test_dataloader(self):
        dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        loader = DataLoader(dataset, **self.dataset_hp)
        return loader

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        acc = self.accuracy(outputs, labels)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
