from typing import List
import os
import logging
import numpy as np
import torch


def get_device(avail_devices: str = None) -> torch.device:
    """
    Returns the least used GPU device if available else MPS or CPU.
    """
    def _get_device(avail_devices: List[int] = None) -> torch.device:
        # get least used gpu by used memory
        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and len(avail_devices) > 0:
            gpu_memory = get_alloc_memory_all_devices()
            gpu_memory = [gpu_memory[i] for i in avail_devices]
            device = torch.device(f'cuda:{avail_devices[np.argmin(gpu_memory)]}')
            return device
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logging.warning("MSP support is still experimental. Use at your own risk!")
                return torch.device("mps")
        except BaseException:
            logging.error("Something went wrong with MPS. Using CPU.")

        return torch.device("cpu")

    # Permanently store the chosen device
    if not hasattr(get_device, 'device'):
        if avail_devices is not None:
            avail_devices = [int(d) for d in avail_devices.split(',')]
        else:
            avail_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible_device is not None:
            avail_devices = [int(d) for d in visible_device.split(',') if d != '' and int(d) in avail_devices]

        get_device.device = _get_device(avail_devices=avail_devices)
        logging.info(f'Using device {get_device.device}')

    return get_device.device


def warn_once(*msg):
    """
    Prints a warning message only once.

    Args:
        msg: the message to be printed
    """
    msg = ' '.join([str(m) for m in msg])
    if not hasattr(warn_once, 'warned'):
        warn_once.warned = set()
    if msg not in warn_once.warned:
        warn_once.warned.add(msg)
        logging.warning(msg)


def is_pynvml_available() -> bool:
    """
    Check whether torch.cuda.pynvml is available and functional.

    Returns:
        bool: True if pynvml via torch.cuda is available and working, False otherwise.
    """
    try:
        if not hasattr(torch.cuda, 'pynvml'):
            return False

        torch.cuda.pynvml.nvmlInit()
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False

        handle = torch.cuda.pynvml.nvmlDeviceGetHandleByIndex(0)
        _ = torch.cuda.pynvml.nvmlDeviceGetMemoryInfo(handle)
        return True

    except Exception:
        return False


def _get_gpu_memory_pynvml_all_processes(device_id: int = 0) -> int:
    """
    Use pynvml to get the memory allocated on the GPU.
    Returns the memory allocated on the GPU in Bytes.
    """
    if not hasattr(_get_gpu_memory_pynvml_all_processes, f'handle_{device_id}'):
        torch.cuda.pynvml.nvmlInit()  # only once
        handle = torch.cuda.pynvml.nvmlDeviceGetHandleByIndex(device_id)
        setattr(_get_gpu_memory_pynvml_all_processes, f'handle_{device_id}', handle)

    handle = getattr(_get_gpu_memory_pynvml_all_processes, f'handle_{device_id}')

    procs = torch.cuda.pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return sum([proc.usedGpuMemory for proc in procs])


def _get_gpu_memory_nvidia_smi() -> List[int]:
    """
    Use nvidia-smi to get current GPU memory usage across all processes.
    Returns memory used in MiB per device, converted to Bytes. Returns [] on failure.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return [int(x.strip()) * 1024 * 1024 for x in result.stdout.strip().split('\n') if x.strip()]
    except Exception:
        pass
    return []


def get_alloc_memory_all_devices(return_all=False) -> List[int]:
    """
    Returns the memory allocated on all the available devices.
    Priority: pynvml > nvidia-smi > torch.max_memory_allocated (cross-process aware order).

    If `return_all` is set to True, it returns a tuple with the memory reserved, allocated and from pynvml.

    Values are in Bytes.
    """
    gpu_memory_reserved = []
    gpu_memory_allocated = []
    gpu_memory_nvidiasmi = []
    for i in range(torch.cuda.device_count()):
        gpu_memory_reserved.append(torch.cuda.max_memory_reserved(i))
        gpu_memory_allocated.append(torch.cuda.max_memory_allocated(i))

        try:
            gpu_memory_nvidiasmi.append(_get_gpu_memory_pynvml_all_processes(i))
        except BaseException as e:
            warn_once('Could not get memory from pynvml. Maybe try `pip install --force-reinstall gpustat`.', str(e))
            gpu_memory_nvidiasmi.append(-1)

    if return_all:
        return gpu_memory_reserved, gpu_memory_allocated, gpu_memory_nvidiasmi
    else:
        # 1순위: pynvml (cross-process, 정확)
        if any([g > 0 for g in gpu_memory_nvidiasmi]):
            return gpu_memory_nvidiasmi
        # 2순위: nvidia-smi subprocess (cross-process, pynvml 없을 때 fallback)
        smi_memory = _get_gpu_memory_nvidia_smi()
        if smi_memory and len(smi_memory) == torch.cuda.device_count():
            return smi_memory
        # 3순위: torch.max_memory_allocated (현재 프로세스 내만 반영, 부정확)
        return gpu_memory_allocated

    
if __name__ == "__main__":
    print(get_device())
    device = get_device()
    print(f"Allocated memory on device {device}: {get_alloc_memory_all_devices()}")