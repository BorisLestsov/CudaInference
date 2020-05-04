import numpy as np
import torch
import torchvision.models as models
from collections import OrderedDict
from sys import argv

#model = models.resnet18(pretrained=True)

model = torch.nn.Sequential(OrderedDict([
    ("conv1", torch.nn.Conv2d(3, 4, 3, bias=True)),
    #("fc1", torch.nn.Linear(3*5*5, 5)),
    #("relu1", torch.nn.ReLU()),
    #("fc2", torch.nn.Linear(5, 10)),
]))

export = False
if export:
    torch.save(model.state_dict(), "fc_net.pth")
else:
    state = torch.load(argv[1])
    model.load_state_dict(state, strict=False)

model.eval()

inp = (np.arange(2*3*5*5).reshape(2, 3, 5, 5))

with torch.no_grad():
    inp = torch.from_numpy(inp).float()
    res = model(inp)

#print(res.shape)
for i, k in enumerate(res.reshape(-1)):
    print(i, k.item())

if export:
    for name, tens in model.named_parameters():
        print(name, tens.shape)
        path = "./weights/" + name
        np_arr = tens.detach().cpu().numpy()
        np.save(path, np_arr)

