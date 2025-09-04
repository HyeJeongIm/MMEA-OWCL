import torch
import random
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device_ids):
    """
    Convert device IDs to torch.device objects
    Args:
        device_ids: list of integers or single integer
    Returns:
        list of torch.device objects
    """
    # Handle single integer input
    if isinstance(device_ids, int):
        device_ids = [device_ids]
    
    # Convert to torch.device objects
    devices = []
    for i in device_ids:
        # Ensure i is an integer
        device_id = int(i) if isinstance(i, str) else i
        if device_id >= 0:
            devices.append(torch.device(f"cuda:{device_id}"))
        else:
            devices.append(torch.device("cpu"))
    
    return devices


def print_args(args):
    for k, v in args.items():
        print(f"{k}: {v}")


def shallow_merge(base: dict, override: dict) -> dict:
    """Shallowly overwrite base with override values."""
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = v
    return out
