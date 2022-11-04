import os
from typing import Union
from torch.nn import Module
import numpy as np
import torch

from analyzer.model import MyModel

cuda_available = torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def get_mask_name(filename: str) -> str:
    filename_wihtout_extension, extension = os.path.splitext(filename)
    return f"{filename_wihtout_extension}_mask.{extension}"


def normalize_imagenet(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0]) / imagenet_stats[1]


def to_best_device(tensor: Union[torch.Tensor, Module]) -> Union[torch.Tensor, Module]:
    if cuda_available:
        tensor = tensor.cuda()
    return tensor


def get_model(eval: bool = False):
    # load the model
    model = MyModel()
    # load the model onto the computation device
    if eval:
        model = model.eval()
    else:
        model = model.train()
    return to_best_device(model)


def do_lod_specific_model(model_path: str, model: Module) -> Module:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model
