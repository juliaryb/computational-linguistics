# config.py
"""
Central configuration for local and Athena runs.
- Auto-detects Athena via $SCRATCH (uses /scratch/<user>/... layout)
- Locally uses ./ as the project base
- Creates needed directories on import
"""

import os
import torch

# -----------------------
# ENV / PATHS
# -----------------------
HOME = os.path.expanduser("~")
SCRATCH_ENV = os.environ.get("SCRATCH")

ON_ATHENA = bool(SCRATCH_ENV) and os.path.isabs(SCRATCH_ENV)

# Base directory for run artifacts (data, tokenizers, logs, checkpoints)
if ON_ATHENA:
    BASE_DIR = os.path.join(SCRATCH_ENV, "computational-linguistics-data")
else:
    BASE_DIR = os.path.abspath("./")  # local project root

DATA_DIR       = os.path.join(BASE_DIR, "data")
TOKENIZER_DIR  = os.path.join(BASE_DIR, "tokenizers")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ensure directories exist
for d in (DATA_DIR, TOKENIZER_DIR, LOGS_DIR, CHECKPOINT_DIR):
    os.makedirs(d, exist_ok=True)

# -----------------------
# DATA FILES
# -----------------------
TRAIN_FILE = "lektury_train.txt"
VALID_FILE = "prose_val.txt"

# Full paths (used elsewhere if needed)
TRAIN_PATH = os.path.join(DATA_DIR, TRAIN_FILE)
VALID_PATH = os.path.join(DATA_DIR, VALID_FILE)

# -----------------------
# TOKENIZER
# -----------------------
# Choose: "spm" (SentencePiece) or "char"
# TOKENIZER_TYPE = "char"  # change to "spm" when you want BPE
TOKENIZER_TYPE = "spm"  # change to "spm" when you want BPE
# Tokenizer model *prefix* (no extension) — both envs share the same variable
# CharTokenizer will use "<prefix>.json"; SentencePiece uses "<prefix>.model"
TOKENIZER_FILE = os.path.join(TOKENIZER_DIR, "sentence-piece") if TOKENIZER_TYPE == "spm" \
                 else os.path.join(TOKENIZER_DIR, "char_tokenizer")

# -----------------------
# MODEL SELECTION
# -----------------------
# Which model to train: "lstm" or "transformer"
MODEL_ARCH = "lstm"
# MODEL_ARCH = "transformer"

# -----------------------
# LSTM HYPERPARAMS
# -----------------------
EMBED_DIM  = 128     # try 128/256 for a stronger baseline
HIDDEN_DIM = 128    # try 256/512 for a stronger baseline
NUM_LAYERS = 2
DROPOUT    = 0.1

# -----------------------
# TRANSFORMER (decoder-only) HYPERPARAMS
# -----------------------
TX_D_MODEL = 128
TX_N_HEAD  = 8
TX_N_LAYER = 2
TX_D_FF    = 256
TX_DROPOUT = 0.1

# -----------------------
# TRAINING
# -----------------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS        = 8
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3

# sequence settings
SEQ_LEN          = 64
DEBUG_MAX_LINES  = 10000  # set to None for full data otherwise takes a subset of corpus

# DataLoader workers: more on Athena, modest locally
if ON_ATHENA:
    NUM_WORKERS = 8
else:
    # keep modest locally; avoid hangs in notebooks
    NUM_WORKERS = max(0, min(4, (os.cpu_count() or 2) - 1))

# -----------------------
# CHECKPOINTS / LOGGING
# -----------------------
MODEL_SAVE_PATH_LSTM = os.path.join(CHECKPOINT_DIR, "best_lstm_model.pth")
MODEL_SAVE_PATH_TX   = os.path.join(CHECKPOINT_DIR, "best_tx_model.pth")
MODEL_SAVE_PATH      = MODEL_SAVE_PATH_LSTM if MODEL_ARCH == "lstm" else MODEL_SAVE_PATH_TX

# one CSV per architecture so you can compare runs easily
LOG_CSV = os.path.join(LOGS_DIR, f"training_metrics_{MODEL_ARCH}.csv")