"""
device.py - decide which chip (PyTorch calls it a "device") does our maths.

Every later script (training, chatting, ...) calls `pick_device()` from here,
so the "which chip?" decision is made in exactly one place.

-----------------------------------------------------------------------------
Plain-English background
-----------------------------------------------------------------------------
CPU vs GPU
    * The CPU (your Ryzen 7) is like a small team of brilliant all-rounders:
      8 cores that can each do anything, one step at a time, very quickly.
    * The GPU (your RX 7800 XT) is like a stadium full of people with
      calculators: thousands of simple workers who all do the same kind of
      sum at the same moment.
    Training a neural network is almost entirely giant grids of
    "multiply these numbers and add them up". That suits the stadium crowd
    perfectly, so for us a GPU is often 10-50 times faster than a CPU.

Why does an AMD card show up as "cuda"?
    CUDA is NVIDIA's system for programming its GPUs, and most AI code in the
    world was written for it. AMD's equivalent is called ROCm (the part that
    talks to programs is called HIP), and it deliberately copies CUDA's
    "shape". So the AMD build of PyTorch reuses all the CUDA names:
    `torch.cuda.is_available()`, `device="cuda"`, and so on all just mean
    "the GPU" - on your PC, the Radeon card. Think of it as a travel adapter:
    the socket is labelled "cuda", but AMD's plug fits it.

Integrated graphics
    Many Ryzen CPUs also contain a tiny GPU of their own ("integrated
    graphics", for driving a monitor without a graphics card). If PyTorch can
    see both, we must use the big RX 7800 XT, not the little one. We do that
    by preferring "not integrated" GPUs, then the one with the most memory
    (VRAM): 16 GB for your card versus a fraction of that for a built-in chip.
"""

import torch

import config


def describe_gpus():
    """Return a list with one small dictionary of facts per GPU PyTorch can see."""
    gpus = []
    if not torch.cuda.is_available():  # True only if PyTorch can use a GPU
        return gpus

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)

        # On AMD, "gcnArchName" is the chip's family code name, e.g. "gfx1101"
        # for the RX 7800 XT. It can carry extra flags after a colon
        # (like "gfx90a:sramecc+:xnack-"), which we chop off.
        # getattr(..., default) means "use the default if this PyTorch build
        # doesn't have that piece of information".
        arch = getattr(props, "gcnArchName", "") or "n/a"

        gpus.append({
            "index": index,                                   # PyTorch's number for it
            "name": props.name,                               # e.g. "AMD Radeon RX 7800 XT"
            "memory_gb": props.total_memory / 1024**3,        # bytes -> gigabytes
            "arch": arch.split(":")[0],                       # e.g. "gfx1101"
            "integrated": bool(getattr(props, "is_integrated", False)),
        })
    return gpus


def _best_gpu(gpus):
    """Prefer a separate (non-integrated) GPU, then the one with the most memory."""
    return max(gpus, key=lambda g: (not g["integrated"], g["memory_gb"]))


def pick_device(verbose=True):
    """
    Choose the device according to config.DEVICE and return it.

    The answer is a `torch.device`, which is PyTorch's way of naming a chip:
    torch.device("cpu") for the CPU, torch.device("cuda:0") for GPU number 0.
    We pass it around so every tensor (grid of numbers) lands on the same chip.
    """
    choice = str(config.DEVICE).strip().lower()
    gpus = describe_gpus()

    if choice == "cpu":
        device, reason = torch.device("cpu"), 'config.DEVICE is "cpu"'

    elif choice == "auto":
        if gpus:
            best = _best_gpu(gpus)
            device = torch.device(f"cuda:{best['index']}")
            reason = "the best GPU PyTorch can see" + (
                f" (picked from {len(gpus)} GPUs)" if len(gpus) > 1 else "")
        else:
            device = torch.device("cpu")
            reason = "PyTorch can't see a usable GPU, so falling back to the CPU"

    elif choice.startswith("cuda"):
        if not gpus:
            raise RuntimeError(
                f'config.DEVICE is "{config.DEVICE}", but PyTorch cannot see a '
                "GPU. Run check_gpu.py to find out why.")
        if choice == "cuda":
            index = _best_gpu(gpus)["index"]
        else:
            index = int(choice.split(":")[1])
            if index >= len(gpus):
                raise RuntimeError(
                    f'config.DEVICE asks for GPU {index}, but PyTorch only sees '
                    f"{len(gpus)} GPU(s), numbered 0 to {len(gpus) - 1}.")
        device = torch.device(f"cuda:{index}")
        reason = f'config.DEVICE is "{config.DEVICE}"'

    else:
        raise ValueError(
            f'config.DEVICE is "{config.DEVICE}"; expected "auto", "cpu", '
            '"cuda" or "cuda:N".')

    if device.type == "cuda":
        # Make this GPU the default one, so any code that just says "cuda"
        # (without a number) also lands on it rather than on GPU 0.
        torch.cuda.set_device(device)

    if verbose:
        print(f"Using device: {device_summary(device)}  <- {reason}")
    return device


def device_summary(device):
    """A short human-readable description of a device, for printing."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        arch = (getattr(props, "gcnArchName", "") or "").split(":")[0]
        arch_text = f", {arch}" if arch else ""
        return f"{device} = {props.name} ({props.total_memory / 1024**3:.1f} GB{arch_text})"
    return f"{device} (the processor)"


def synchronize(device):
    """
    Wait until the device has really finished its work.

    GPUs work "asynchronously": Python hands the GPU a job and immediately
    moves on, like dropping a letter in a postbox. That's great for speed,
    but if we want to time something we must wait for the GPU to actually
    finish, or we'd only be timing how long it took to post the letter.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
