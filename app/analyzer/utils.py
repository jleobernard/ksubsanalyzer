from typing import Union

import torch
from torch.nn import Module

cuda_available = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def to_best_device(tensor: Union[torch.Tensor, Module]) -> Union[torch.Tensor, Module]:
    if cuda_available:
        tensor = tensor.cuda()
    return tensor

