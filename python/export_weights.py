import numpy as np
import torch
import torchvision.models as models
from collections import OrderedDict
from sys import argv
from operator import attrgetter

export = False

with torch.no_grad():
    model = torch.nn.Sequential(OrderedDict([
        #("conv1", torch.nn.Conv2d(3, 4, 3, stride=2, padding=2, bias=True)),
        ("avgp", torch.nn.AvgPool2d(5, stride=1, padding=0)),
        #("fc1", torch.nn.Linear(3*5*5, 5)),
        #("relu1", torch.nn.ReLU()),
        #("fc2", torch.nn.Linear(5, 10)),
    ]))


# model = models.resnet18(pretrained=True)
# print(model)
if export:
    for modn, mod in model.named_modules():
        tensors = []
        print("CHECK", modn)
        if 'bn' in modn:
            tensors.append((modn+'.running_mean', mod.running_mean))
            tensors.append((modn+'.running_var', mod.running_var))
        tensors += [(modn+'.'+k, v) for k, v in mod.named_parameters()]
        for name, tens in tensors:
            print(name, tens.shape)
            path = "./weights/" + name
            np_arr = tens.detach().cpu().numpy()
            np.save(path, np_arr)

if export:
    torch.save(model.state_dict(), './weights/' + "fc_net.pth")
    exit()
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

