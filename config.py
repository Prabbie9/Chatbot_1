"""
config.py - the "control panel" for ChatGDP.

Every setting you might want to change lives in this one file, so you never
have to hunt through the code for "magic numbers". Other files simply do
`import config` and read what they need, e.g. `config.BOT_NAME`.

We'll add more settings as we go: data locations in Stage 1, tokenizer size in
Stage 2, model size in Stage 3, training settings in Stage 4, and so on.
"""

# =============================================================================
# 1. The assistant's name
# =============================================================================
# This is the ONLY place the name "ChatGDP" is written down. Everything else
# (the "who are you?" practice conversations we create in Stage 1, the web
# page in Stage 6) reads it from here.
#
# What would changing it involve? Edit the line below, then:
#   * Pretraining (Stage 4, the overnight run): nothing to redo. That stage
#     only teaches the model general English, and it never sees the name.
#   * Tokenizer (Stage 2): nothing to redo. It works on raw bytes, so it can
#     spell any name. (A new name may just be split into a few more pieces.)
#   * Chat data (Stage 1): re-run the data script, so the self-introduction
#     examples use the new name.
#   * Chat tuning (Stage 5): re-run it (much shorter than pretraining), because
#     that is where the model actually *learns* what it is called.
#   * Web page (Stage 6): nothing; it reads the name from here automatically.
# The project folder can stay called "chatgdp"; the folder name doesn't matter.
BOT_NAME = "ChatGDP"


# =============================================================================
# 2. Which chip does the maths? (see device.py for the full explanation)
# =============================================================================
#   "auto"   -> use the best GPU PyTorch can find, i.e. your RX 7800 XT rather
#               than the small graphics chip built into some Ryzen CPUs.
#               If no GPU works at all, fall back to the CPU (slow but safe).
#   "cuda"   -> insist on the best GPU, and stop with an error if there isn't
#               one. (Yes, "cuda" even on AMD - device.py explains why.)
#   "cuda:1" -> insist on one particular GPU, by its number.
#   "cpu"    -> always use the CPU.
DEVICE = "auto"
