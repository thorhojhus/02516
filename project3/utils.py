import torch
import numpy as np
import random

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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