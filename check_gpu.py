r"""
check_gpu.py - the Stage 0 health check.

Question: is PyTorch installed properly, and can it use your RX 7800 XT?

Run it from inside the project folder:
    .venv\Scripts\python check_gpu.py

It only *reads* information and does some sums; it changes nothing on your PC.
A copy of everything it prints is saved to logs\check_gpu_report.txt, so it's
easy to share.

The six checks:
  1. Your computer  - Python version, Windows version, graphics chips
  2. PyTorch        - is it installed, and is it AMD's ROCm version?
  3. GPUs           - which GPUs PyTorch can see, and which one we'll use
  4. Correctness    - does the GPU get the same answers as the CPU?
  5. Speed          - how many trillion sums per second can each chip do?
  6. Mini training  - can the GPU train a tiny neural network?
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_FILE = PROJECT_DIR / "logs" / "check_gpu_report.txt"

REPORT_LINES = []  # everything we print, so we can also save it to a file
RESULTS = []       # (status, message) pairs, repeated in the summary at the end


# -----------------------------------------------------------------------------
# Small printing helpers
# -----------------------------------------------------------------------------
def say(text=""):
    """Print a line AND remember it for the saved report."""
    print(text, flush=True)
    REPORT_LINES.append(text)


def heading(title):
    say()
    say("=" * 72)
    say(title)
    say("=" * 72)


def record(status, message):
    """Note a result. status is "OK" (good), "WARN" (worth a look) or "FAIL"."""
    RESULTS.append((status, message))
    say(f"[{status}] {message}")


# -----------------------------------------------------------------------------
# 1. Your computer
# -----------------------------------------------------------------------------
# Windows 11 "build numbers" and the friendly names Microsoft gives them.
WINDOWS_11_BUILDS = {22000: "21H2", 22621: "22H2", 22631: "23H2",
                     26100: "24H2", 26200: "25H2"}


def describe_windows():
    """Turn the Windows build number into a friendly name like "Windows 11 25H2"."""
    version = platform.version()  # e.g. "10.0.26200" (Windows 11 still says "10.0"!)
    try:
        build = int(version.split(".")[2])
    except (IndexError, ValueError):
        return f"Windows (version {version})", None
    if build < 22000:
        return f"Windows 10 or older (build {build})", build
    known = [b for b in sorted(WINDOWS_11_BUILDS) if b <= build]
    name = WINDOWS_11_BUILDS[known[-1]] + ("" if known[-1] == build else " or newer")
    return f"Windows 11 {name} (build {build})", build


def windows_hardware():
    """
    Ask Windows (through PowerShell) for the processor name and the list of
    graphics chips it knows about. This is read-only.
    Returns (cpu_name, [(gpu_name, driver_version, driver_date), ...]).
    """
    script = (
        "$ErrorActionPreference = 'SilentlyContinue';"
        "$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name;"
        "Write-Output ('CPU|' + $cpu);"
        "Get-CimInstance Win32_VideoController | ForEach-Object {"
        "  $d = if ($_.DriverDate) { $_.DriverDate.ToString('yyyy-MM-dd') } else { 'unknown' };"
        "  Write-Output ('GPU|' + $_.Name + '|' + $_.DriverVersion + '|' + $d) }"
    )
    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, errors="replace", timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, []
    cpu, gpus = None, []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if parts[0] == "CPU" and len(parts) >= 2:
            cpu = parts[1]
        elif parts[0] == "GPU" and len(parts) >= 4:
            gpus.append((parts[1], parts[2], parts[3]))
    return cpu, gpus


def linux_cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def check_computer():
    heading("1. Your computer")

    # --- Python itself ---
    version = ".".join(map(str, sys.version_info[:3]))
    bits = "64-bit" if sys.maxsize > 2**32 else "32-bit"
    say(f"Python: {version} ({bits}) at {sys.executable}")
    if (3, 11) <= sys.version_info[:2] <= (3, 14):
        record("OK", f"Python {version} is one AMD builds PyTorch for (3.11 to 3.14).")
    else:
        record("FAIL", f"Python {version}: AMD's Windows PyTorch needs Python 3.11 to "
                       "3.14 (3.12 recommended). See 'What you need first' in the README.")
    if bits != "64-bit":
        record("FAIL", "This is 32-bit Python; PyTorch needs 64-bit Python.")

    # A virtual environment is a separate copy of Python's add-ons for this
    # project. If we're inside one, sys.prefix (where packages live) differs
    # from sys.base_prefix (the main Python installation).
    if sys.prefix != sys.base_prefix:
        record("OK", f"Running inside a virtual environment ({sys.prefix}).")
    else:
        record("WARN", "Not running inside the project's virtual environment. Use "
                       r"'.venv\Scripts\python check_gpu.py' (see README).")

    # --- Operating system and hardware ---
    if platform.system() == "Windows":
        os_name, build = describe_windows()
        say(f"Operating system: {os_name}")
        if build is not None and build >= 26200:
            record("OK", "Windows 11 25H2 or newer, which AMD lists for ROCm 10.1.")
        elif build is not None and build >= 22000:
            record("WARN", "AMD only lists Windows 11 25H2 for ROCm 10.1. Older "
                           "Windows 11 may still work; if the GPU isn't found, "
                           "a Windows update is one fix to consider.")
        else:
            record("WARN", "AMD lists only Windows 11 (25H2) for ROCm 10.1; "
                           "Windows 10 is not on AMD's list.")

        cpu, adapters = windows_hardware()
        say(f"Processor: {cpu or 'unknown'}")
        if adapters:
            say("Graphics chips Windows knows about:")
            for name, driver, date in adapters:
                # A plain "Radeon(TM) Graphics" (no "RX" model number) is
                # usually the small graphics chip built into a Ryzen CPU.
                is_builtin = "radeon" in name.lower() and " RX " not in f" {name} "
                hint = "   <- probably the graphics built into the CPU" if is_builtin else ""
                say(f"  - {name}  (Windows driver {driver}, dated {date}){hint}")
            say("Tip: the AMD Software app (System tab) shows the friendlier "
                "'Adrenalin' driver number; AMD lists 26.10.41.05 for ROCm 10.1.")
    else:
        say(f"Operating system: {platform.system()} {platform.release()}")
        say(f"Processor: {linux_cpu_name()}")
        say("(Not Windows: fine for testing the code, but the AMD install steps "
            "in the README are the Windows ones.)")

    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(var):
            say(f"Note: environment variable {var}={os.environ[var]} limits "
                "which GPUs PyTorch can see.")


# -----------------------------------------------------------------------------
# 2. PyTorch
# -----------------------------------------------------------------------------
def check_pytorch():
    heading("2. PyTorch")
    try:
        import torch
    except ImportError:
        record("FAIL", "PyTorch isn't installed in this Python. Follow README, "
                       "Stage 0, step 3.")
        return None

    say(f"PyTorch version: {torch.__version__}")
    hip = getattr(torch.version, "hip", None)    # set on AMD ROCm builds
    cuda = getattr(torch.version, "cuda", None)  # set on NVIDIA CUDA builds
    if hip:
        record("OK", f"This is AMD's ROCm build of PyTorch (HIP {hip}).")
    elif cuda:
        record("FAIL", f"This is an NVIDIA (CUDA {cuda}) build of PyTorch; it can't "
                       "drive an AMD card. Reinstall from AMD's server (README step 3).")
    else:
        record("FAIL", "This is a CPU-only build of PyTorch, so it can't use any "
                       "GPU. That's what a plain 'pip install torch' gives you on "
                       "Windows. Reinstall from AMD's server (README step 3).")
    return torch


# -----------------------------------------------------------------------------
# 3. GPUs
# -----------------------------------------------------------------------------
def check_gpus(torch):
    heading("3. GPUs that PyTorch can see")
    from device import describe_gpus, device_summary, pick_device

    say("(The first time the GPU is used it can pause for up to a minute while "
        "the driver loads. That's normal.)")
    try:
        gpus = describe_gpus()
    except Exception as error:  # a broken driver can raise here
        record("FAIL", f"Asking PyTorch about GPUs crashed: {error!r}")
        gpus = []

    if not gpus:
        say("PyTorch can't see any GPU.")
        if getattr(torch.version, "hip", None):
            say("You have the AMD build of PyTorch, so the usual causes are:")
            say("  - the graphics driver is older than ROCm 10.1 needs "
                "(AMD lists Adrenalin 26.10.41.05), or")
            say("  - Windows is older than AMD supports (Windows 11 25H2 is listed).")
        record("FAIL", "No usable GPU, so everything would run on the (much "
                       "slower) CPU. See 'If something goes wrong' in README.md.")
    else:
        for g in gpus:
            kind = "built-in (integrated)" if g["integrated"] else "separate card"
            say(f"  GPU {g['index']}: {g['name']} | chip {g['arch']} | "
                f"{g['memory_gb']:.1f} GB memory | {kind}")
        if len(gpus) > 1:
            say("More than one GPU found (often the Ryzen's built-in graphics plus "
                "the RX 7800 XT). device.py picks the separate card with the most "
                "memory.")

    try:
        device = pick_device(verbose=False)
    except Exception as error:
        record("FAIL", f"Couldn't choose a device: {error}")
        return None
    say(f"ChatGDP will use: {device_summary(device)}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        arch = (getattr(props, "gcnArchName", "") or "").split(":")[0]
        if arch == "gfx1101":
            record("OK", f"Using {props.name}: chip family gfx1101, which is "
                         "the RX 7800 XT's.")
        else:
            record("WARN", f"Using {props.name} (chip {arch or 'unknown'}); an RX "
                           "7800 XT should report gfx1101.")
    return device


# -----------------------------------------------------------------------------
# 4. Correctness
# -----------------------------------------------------------------------------
def check_correctness(torch, device):
    heading("4. Correctness: does the GPU get the same answers as the CPU?")
    if device.type != "cuda":
        say("Skipped: there's no GPU to compare against.")
        return
    try:
        torch.manual_seed(0)
        # Matrix multiplication: the single most important operation in a
        # neural network. Two 512x512 grids of random numbers.
        a, b = torch.randn(512, 512), torch.randn(512, 512)
        on_cpu = a @ b
        on_gpu = (a.to(device) @ b.to(device)).cpu()  # .to() moves data to the GPU
        matmul_error = ((on_cpu - on_gpu).abs().max() / on_cpu.abs().max()).item()

        # Softmax: turns a list of scores into percentages that add up to 100%.
        # (We'll meet it again in Stage 3.)
        x = torch.randn(64, 1000)
        softmax_error = (torch.softmax(x, dim=-1)
                         - torch.softmax(x.to(device), dim=-1).cpu()).abs().max().item()

        say(f"Matrix multiply: biggest difference CPU vs GPU = {matmul_error:.1e} "
            "(relative)")
        say(f"Softmax:         biggest difference CPU vs GPU = {softmax_error:.1e}")
        say("(Tiny differences are normal: the chips add numbers up in different "
            "orders, and rounding differs very slightly.)")
        if matmul_error < 1e-3 and softmax_error < 1e-5:
            record("OK", "GPU and CPU agree.")
        else:
            record("FAIL", "GPU answers differ from the CPU by more than rounding "
                           "can explain.")
    except Exception as error:
        record("FAIL", f"The correctness test crashed: {error!r}")


# -----------------------------------------------------------------------------
# 5. Speed
# -----------------------------------------------------------------------------
def matmul_tflops(torch, device, dtype, size, repeats):
    """
    Time `repeats` multiplications of two size x size grids and return the
    speed in TFLOPS (trillions of floating-point operations per second).
    Multiplying two n x n grids takes about 2 * n^3 operations.
    """
    from device import synchronize
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(3):  # warm-up: the first runs include one-off setup costs
        a @ b
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    synchronize(device)  # wait for the GPU to really finish (see device.py)
    seconds = time.perf_counter() - start
    return 2 * size**3 * repeats / seconds / 1e12


def check_speed(torch, device):
    heading("5. Speed: how many trillion sums per second?")
    say("We time big matrix multiplications, the core job of training. Speeds "
        "are in TFLOPS = trillions of maths operations per second.")
    say("fp32 = numbers stored in 32 bits (very precise). fp16 / bf16 = 16 bits:")
    say("half the memory and much faster on modern GPUs, slightly less precise.")
    say()

    cpu = torch.device("cpu")
    try:
        cpu_speed = matmul_tflops(torch, cpu, torch.float32, size=2048, repeats=3)
        say(f"  CPU, fp32:  {cpu_speed:7.2f} TFLOPS")
    except Exception as error:
        cpu_speed = None
        say(f"  CPU test crashed: {error!r}")

    if device.type != "cuda":
        say("  (No GPU to test.)")
        return

    best = 0.0
    for label, dtype, repeats in (("fp32", torch.float32, 10),
                                  ("fp16", torch.float16, 20),
                                  ("bf16", torch.bfloat16, 20)):
        try:
            speed = matmul_tflops(torch, device, dtype, size=4096, repeats=repeats)
            best = max(best, speed)
            say(f"  GPU, {label}:  {speed:7.2f} TFLOPS")
        except Exception as error:
            record("WARN", f"GPU {label} test crashed: {error!r}")

    if best > 0:
        if cpu_speed:
            say(f"\nThe GPU's best speed is about {best / cpu_speed:.0f}x the CPU's.")
        record("OK", f"GPU speed measured: best {best:.1f} TFLOPS.")


# -----------------------------------------------------------------------------
# 6. Mini training
# -----------------------------------------------------------------------------
def check_training(torch, device):
    heading("6. Mini training: can this chip train a tiny neural network?")
    say("Toy task: 4,096 random dots in a square; the network must learn which "
        "dots are inside a circle.")
    say("'Loss' is a score for how wrong its guesses are (lower is better).")
    say("Training means: guess, measure the loss, nudge the network's numbers "
        "to make the loss smaller, repeat.")
    say("We'll do exactly the same thing to train ChatGDP, just much bigger.\n")
    try:
        torch.manual_seed(0)
        dots = torch.rand(4096, 2, device=device) * 2 - 1          # x, y in [-1, 1]
        inside = ((dots ** 2).sum(dim=1) < 0.5).float()            # 1 = inside circle

        # A tiny network: 2 numbers in -> 64 -> 64 -> 1 number out.
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)  # the "nudger"
        loss_fn = torch.nn.BCEWithLogitsLoss()  # loss for yes/no questions

        start = time.perf_counter()
        for step in range(301):
            guesses = net(dots).squeeze(1)      # 1. guess
            loss = loss_fn(guesses, inside)     # 2. measure how wrong
            optimizer.zero_grad()
            loss.backward()                     # 3. work out which way to nudge
            optimizer.step()                    # 4. nudge
            if step % 50 == 0:
                accuracy = ((guesses > 0).float() == inside).float().mean().item()
                say(f"  step {step:3d}   loss {loss.item():.3f}   "
                    f"right answers {accuracy:6.1%}")
        seconds = time.perf_counter() - start

        if loss.item() < 0.2 and accuracy > 0.93:
            record("OK", f"Mini training worked on {device} in {seconds:.1f} s "
                         "(loss went down, accuracy went up).")
        else:
            record("FAIL", "Mini training didn't learn as expected.")
    except Exception as error:
        record("FAIL", f"Mini training crashed: {error!r}")


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def finish(device):
    heading("Summary")
    for status, message in RESULTS:
        say(f"[{status}] {message}")
    failures = sum(status == "FAIL" for status, _ in RESULTS)
    say()
    if device is not None and device.type == "cuda" and failures == 0:
        say("RESULT: Your GPU is working with PyTorch. Stage 0 is complete!")
    elif device is not None and failures == 0:
        say("RESULT: PyTorch works, but only on the CPU.")
    else:
        say("RESULT: Something needs fixing. See the [FAIL] lines above and "
            "'If something goes wrong' in README.md.")

    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text("\n".join(REPORT_LINES) + "\n", encoding="utf-8")
    print(f"\n(A copy of this report was saved to {REPORT_FILE})")
    return 1 if failures else 0


def main():
    say("ChatGDP - Stage 0 GPU check")
    check_computer()
    torch = check_pytorch()
    device = None
    if torch is not None:
        device = check_gpus(torch)
        if device is not None:
            check_correctness(torch, device)
            check_speed(torch, device)
            check_training(torch, device)
    return finish(device)


if __name__ == "__main__":
    sys.exit(main())
