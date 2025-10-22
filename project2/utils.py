import torch
import numpy as np
import random
from project2.plotting import *

def set_seed(seed: int):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

def get_cuda_device_arch():
    if not torch.cuda.is_available():
        return None
    device = torch.device('cuda')
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor

def set_default_dtype_based_on_arch():
    arch = get_cuda_device_arch()
    if arch is None:
        torch.set_default_dtype(torch.float32)
    if arch >= 80:
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float16)