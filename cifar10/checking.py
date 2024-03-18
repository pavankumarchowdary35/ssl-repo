import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import sys
import argparse
import os
import time
from torch.nn.parallel import DataParallel


from dataset.cifar10 import get_dataset

sys.path.append('../utils_pseudoLab/')
from TwoSampler import *
from utils_ssl import *
from utils_ssl import loss_soft_reg_ep

from ssl_networks import CNN as MT_Net
from PreResNet import PreactResNet18_WNdrop
from wideArchitectures import WRN28_5_wn
import models_teacher.wideresnet as wrn_models

def create_teacher_model():
        
    print("==> creating WideResNet" + str(28) + '-' + str(5))
    model = wrn_models.WideResNet(first_stride =  1,
                                            num_classes  = 10,
                                            depth        = 28,
                                            widen_factor = 5,
                                            activation   = 'relu')

    return model


transform_test = transforms.Compose([
        transforms.ToTensor()])

model_teacher = create_teacher_model()
checkpoint = torch.load('checkpoint_paper/best.pth.tar')
model_teacher.load_state_dict(checkpoint['state_dict'])


path = "./checkpoints/trades/mixup/warmUp_Mdrop1_1_20_cifar10_4000_WRN28_5_wn_S501.hdf5"
checkpoint = torch.load(path)
model = WRN28_5_wn(num_classes = 10, dropout = 0.0)

state_dict = checkpoint['state_dict']

new_state_dict = state_dict.copy()

# Remove "module." prefix from keys if present (for models trained with DataParallel)
new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
model.load_state_dict(new_state_dict)




# model.load_state_dict(checkpoint['state_dict'])
print("Load model in epoch " + str(checkpoint['epoch']))
print("Path loaded: ", path)


testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)

def calculate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images
            labels = labels
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# Usage
accuracy_teacher = calculate_accuracy(model_teacher, test_loader)
print("Accuracy of model_teacher on the test set: {:.2f}%".format(accuracy_teacher * 100))

accuracy_model = calculate_accuracy(model, test_loader)
print("Accuracy of model on the test set: {:.2f}%".format(accuracy_model * 100))