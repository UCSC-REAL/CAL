from inspect import isclass, isfunction

from networks.resnet_cifar import *



adict = {k:v for k, v in locals().items()}

alllist = []
skiplist = ["isclass", "isfunction"]
for k, v in adict.items():
    if isclass(v):
        alllist.append(v.__name__)
    if isfunction(v):
        if v.__name__ not in skiplist:
            alllist.append(v.__name__)

__all__ = alllist