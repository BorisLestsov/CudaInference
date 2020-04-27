import numpy as np
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

for name, tens in model.named_parameters():
    print(name, tens.shape)
