import numpy as np
import torch
import torchvision.models as models
from collections import OrderedDict
from sys import argv

#model = models.resnet18(pretrained=True)

model = torch.nn.Sequential(OrderedDict([
    ("fc1", torch.nn.Linear(3*5*5, 5)),
    ("relu1", torch.nn.ReLU()),
    ("fc2", torch.nn.Linear(5, 10)),
]))

#torch.save(model.state_dict(), "fc_net.pth")
state = torch.load(argv[1])
model.load_state_dict(state, strict=False)

model.eval()

w = model.fc1.weight.detach().cpu().numpy()
inp = (np.arange(1*3*5*5).reshape(1, 3*5*5) + 1) % (3*5*5)

with torch.no_grad():
    inp = torch.from_numpy(inp).float()
    res = model(inp)
    print(res)

if False:
    for name, tens in model.named_parameters():
        print(name, tens.shape)
        path = "./weights/" + name
        np_arr = tens.detach().cpu().numpy()
        np.save(path, np_arr)

