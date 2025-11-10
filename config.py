# config.py
"""
Central configuration file for all hyperparameters and settings.
"""

import torch

# --- Dirs ---
DATA_DIR = "./data" # Directory where your .txt files are
TRAIN_FILE = "lektury_train.txt" # Your training data file
VALID_FILE = "prose_val.txt" # Your validation data file

MODEL_SAVE_PATH = "best_lstm_model.pth" # Path to save the best model

TOKENIZER_TYPE = "spm" # or "char"
# TOKENIZER_TYPE = "char"
TOKENIZER_FILE = "sentence-piece" # this is the model prefix parameter during creation of sentencpiece
# TOKENIZER_FILE = "char_tokenizer" # Path to save/load tokenizer

# --- Data ---
# For a "smoke test", set SEQ_LEN to 32 and DEBUG_MAX_LINES to 1000
# For a real run, set SEQ_LEN to 128 or 256 and DEBUG_MAX_LINES to None
SEQ_LEN = 128       # How many tokens in one sequence (sample)
DEBUG_MAX_LINES = 1000 # Set to e.g., 1000 to load only 1k lines for a quick test

# --- Model ---
# For a "smoke test", use small values
# For a real run, increase these
EMBED_DIM = 128      # Dimension for token embeddings
HIDDEN_DIM = 256     # Dimension for LSTM hidden states
NUM_LAYERS = 2       # Number of LSTM layers
DROPOUT = 0.1        # Dropout probability

# --- Training ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# For a "smoke test", set EPOCHS to 1 and BATCH_SIZE to 8
# For a real run, increase these
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
