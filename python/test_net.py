import numpy as np
import PIL.Image as Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

model = models.resnet18(pretrained=True)
model.eval().cuda()

tr = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]
)

img = Image.open("../images/220px-Lenna_test_image.PPM")

tens = tr(img).unsqueeze(0)
tens = tens.cuda()

out = model(tens)

print(out)
