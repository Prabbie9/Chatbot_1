# ChatGDP: a tiny chatbot built from scratch

ChatGDP is a small chat assistant built **completely from scratch**, one piece
at a time, to understand how chatbots like ChatGPT really work. We write our
own tokenizer, transformer model, training loop and text generator, plus a
small web page for chatting with it. Everything runs on one home PC (Windows,
AMD Radeon RX 7800 XT).

**Honest expectations:** ChatGDP will be tiny: tens of millions of
*parameters* (the adjustable numbers inside the model), against hundreds of
billions for the big chatbots. It will write simple, child-like English and
manage short, simple conversations, but it will often be silly, forgetful or
wrong. That's expected. The goal is understanding, not beating ChatGPT.

## The plan

| Stage | What we build | Rough time | Status |
|---|---|---|---|
| 0 | Setup: Python, virtual environment, PyTorch on the AMD GPU | 30-60 min | **In progress** |
| 1 | Data: simple stories, simple conversations, our chat format | ~1 hour | |
| 2 | Tokenizer: turning text into numbers (byte-level BPE) | 1-2 hours | |
| 3 | Model: a small GPT-style transformer, plus speed tests | 1-2 hours | |
| 4 | Pretraining: learning English overnight | 1 hour + one night | |
| 5 | Chat tuning: teaching it to answer as an assistant | 1-2 hours | |
| 6 | Web chat page in your browser | 1-2 hours | |
| 7 | Wrap-up: how it all fits together, and next steps | ~30 min | |

## How we're working

Claude writes and tests the code in a cloud workspace and saves it to this
GitHub repository. You download it to your PC with Git and run it there,
because the training has to happen on your own graphics card. At the start of
each new stage, run `git pull` in the project folder to fetch the new files.

## What's in this folder

| File | What it does |
|---|---|
| `config.py` | The control panel: every setting in one place, including the bot's name |
| `device.py` | Picks which chip does the maths (the RX 7800 XT, not the CPU's built-in graphics) |
| `check_gpu.py` | Stage 0 health check: is PyTorch installed, and is it using the GPU? |
| `requirements-rocm-windows.txt` | The exact PyTorch build AMD makes for your card |
| `requirements.txt` | Other Python packages we use, from the normal Python package index |
| `.gitignore` | Tells Git which big or personal files never to upload |

---

## Stage 0: set up your PC

### Words you'll meet

- **Terminal (PowerShell):** a window where you type commands. Right-click
  the Start button and choose *Terminal*.
- **Python:** the programming language we write ChatGDP in.
- **Virtual environment ("venv"):** a private toolbox for one project. It's a
  folder called `.venv` that holds this project's own copy of Python's add-on
  packages. Installing things for ChatGDP can't break any other Python program
  on your PC, and deleting the folder removes them all cleanly.
- **pip:** Python's installer for add-on *packages*.
- **PyTorch:** a free library that does the heavy maths for us: grids of
  numbers (*tensors*), automatic gradients, optimisers. We write the chatbot
  ourselves; PyTorch is our calculator.
- **ROCm:** AMD's software that lets programs like PyTorch use Radeon GPUs.

### What you need first

| Need | How to check |
|---|---|
| Windows 11 (AMD lists version 25H2) | Press Win+R, type `winver`, press Enter |
| AMD driver for ROCm 10.1 (AMD lists *Adrenalin Edition 26.10.41.05*) | Open AMD Software: Adrenalin Edition, then the *System* tab |
| Python 3.11, 3.12, 3.13 or 3.14 (3.12 recommended) | `py --list` (or `python --version`) in a terminal |
| Git | `git --version` in a terminal |
| About 15 GB of free disk space | File Explorer, *This PC* |

If anything is missing, stop here. Installing drivers, Python or Git changes
your whole system, so we'll decide on each one together first.

### Step 1: get the project

Pick a short folder path that isn't synced by OneDrive, such as
`C:\Projects\chatgdp`. Short paths avoid Windows' 260-character path limit,
and OneDrive syncing gigabytes of training files would be slow and can lock
files mid-save. (If Git asks you to sign in to GitHub, that's normal.)

```powershell
git clone --branch claude/vigilant-goodall-t8qikm https://github.com/prabbie9/chatbot_1.git C:\Projects\chatgdp
```

```powershell
cd C:\Projects\chatgdp
```

### Step 2: create the virtual environment

Use your Python version in place of `3.12` if it's different:

```powershell
py -3.12 -m venv .venv
```

This creates the `.venv` folder. From now on, we always run
`.venv\Scripts\python`, which is the project's private Python, rather than
plain `python`.

### Step 3: install PyTorch for your AMD GPU

First update pip itself, then install AMD's PyTorch (about 1.6 GB to download,
so this can take a while):

```powershell
.venv\Scripts\python -m pip install --upgrade pip
```

```powershell
.venv\Scripts\python -m pip install -r requirements-rocm-windows.txt
```

You'll see a line like `Building wheel for rocm`. That's normal: it's a small
AMD helper package being prepared on your PC.

### Step 4: install the other packages

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

### Step 5: run the GPU check

```powershell
.venv\Scripts\python check_gpu.py
```

It prints a report and saves a copy to `logs\check_gpu_report.txt`. The last
line should read: `RESULT: Your GPU is working with PyTorch. Stage 0 is complete!`

### Optional: "activating" the virtual environment

Activating a venv lets you type `python` instead of `.venv\Scripts\python`.
In PowerShell the command is `.venv\Scripts\Activate.ps1`, but Windows often
blocks it with *"running scripts is disabled on this system"*. Running
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first lifts that
block for the current terminal window only, and it resets when you close the
window. Typing `.venv\Scripts\python` avoids the question entirely, which is
why this guide does that.

### If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `'py' is not recognized` | Python isn't installed, or the `py` launcher wasn't included | Try `python --version`; otherwise we'll install Python together |
| `No matching distribution found for torch...` | Your Python isn't a version AMD builds for (3.11 to 3.14, 64-bit) | Recreate `.venv` with a supported Python |
| Install fails mentioning a missing file or a long path | The project folder's path is too long | Move the project to a short path like `C:\Projects\chatgdp` |
| check_gpu says *CPU-only build of PyTorch* | PyTorch came from the normal index (a plain `pip install torch`) | `.venv\Scripts\python -m pip uninstall -y torch`, then repeat step 3 |
| check_gpu says *PyTorch can't see any GPU* | Usually the driver (or Windows) is older than ROCm 10.1 needs | Compare your driver and Windows versions with the table above |
| check_gpu picks the built-in graphics | Rare: the selection rule guessed wrong | Set `DEVICE = "cuda:1"` in `config.py`, using the RX 7800 XT's GPU number from the report |

### If the GPU won't work: fallbacks

We only go down this road if a reasonable attempt with AMD's official route
fails.

- **WSL2:** a Linux system running inside Windows. AMD's ROCm 10.1 supports it
  as a "technical preview". Installing WSL is a system-wide change.
- **DirectML (`torch-directml`):** Microsoft's GPU route that works on most
  graphics cards. Its last release was in September 2024, so it's stuck on an
  old PyTorch (2.4.1) and Python 3.12 or older, and some operations are
  missing or slow.
- **CPU only:** always works, but much slower. `check_gpu.py` measures exactly
  how much slower on your PC. We'd shrink the model to a few million
  parameters and train on less text.

---

## Changing the bot's name

The name lives in one place: `BOT_NAME` in `config.py`. To rename the bot:

1. Edit `BOT_NAME`.
2. Re-run the Stage 1 data script, so the "Who are you?" examples use the new name.
3. Re-run Stage 5 chat tuning, a much shorter run than pretraining. That's
   where the model actually learns what it's called.

You don't need to redo the tokenizer (it can spell any name) or the overnight
pretraining run (it never sees the name). The web page picks up the new name
automatically.
