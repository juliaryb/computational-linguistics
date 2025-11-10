"""
Main training script for the LSTM Language Model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import math

# Import from our other files
import config
from tokenizer import ensure_char_tokenizer, ensure_spm_tokenizer
from data import LanguageModelDataset
from model import LSTMModel

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Runs one full epoch of training."""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Initialize hidden state at the beginning of the epoch
    # We detach it to prevent backpropping through all of time
    hidden = model.init_hidden(config.BATCH_SIZE, device)

    for i, (x, y) in enumerate(dataloader):
        # Ensure batch size is consistent, drop last batch if it's smaller
        if x.shape[0] != config.BATCH_SIZE:
            continue
            
        x, y = x.to(device), y.to(device)
        
        # Detach the hidden state from the previous batch's history
        hidden = (hidden[0].detach(), hidden[1].detach())
        
        # --- Forward pass ---
        optimizer.zero_grad()
        logits, hidden = model(x, hidden)
        
        # --- Compute Loss ---
        # CrossEntropyLoss expects logits as (N, C) and labels as (N)
        # We flatten the (batch_size, seq_len, vocab_size) and (batch_size, seq_len) tensors
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
        
        # --- Backward pass and optimization ---
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    print(f"  > Train Epoch Time: {elapsed:.2f}s")
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    """Runs evaluation on the validation set."""
    model.eval()
    total_loss = 0
    
    hidden = model.init_hidden(config.BATCH_SIZE, device)

    with torch.no_grad(): # disable gradient calculation
        for x, y in dataloader:
            if x.shape[0] != config.BATCH_SIZE:
                continue
                
            x, y = x.to(device), y.to(device)
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            logits, hidden = model(x, hidden)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) # calculate perplexity
    return avg_loss, perplexity


def main():
    print(f"Using device: {config.DEVICE}")
    
    # --- 1. Load Tokenizer ---
    train_data_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    tokenizer_path = os.path.join(config.DATA_DIR, config.TOKENIZER_FILE)
    
    # trains a tokenizer if it doesn't exist
    if config.TOKENIZER_TYPE == "char":
        tokenizer = ensure_char_tokenizer(train_data_path, tokenizer_path)
    elif config.TOKENIZER_TYPE == "spm":
        tokenizer = ensure_spm_tokenizer(train_data_path, tokenizer_path)
    else:
        raise ValueError(f"Unknown TOKENIZER_TYPE={config.TOKENIZER_TYPE}")
    
    vocab_size = tokenizer.get_vocab_size()
    
    # --- 2. Load Data ---
    valid_data_path = os.path.join(config.DATA_DIR, config.VALID_FILE)
    
    train_dataset = LanguageModelDataset(train_data_path, tokenizer, config.SEQ_LEN, config.DEBUG_MAX_LINES)
    valid_dataset = LanguageModelDataset(valid_data_path, tokenizer, config.SEQ_LEN, config.DEBUG_MAX_LINES)
    
    # drop_last=True is important to ensure all batches have config.BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=True)

    # --- 3. Initialize Model ---
    model = LSTMModel(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        tokenizer=tokenizer
    ).to(config.DEVICE)
    
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 4. Training Setup ---
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_perplexity = float("inf")
    
    # --- 5. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{config.EPOCHS}]")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        print(f"  > End of Epoch, Train Loss: {train_loss:.4f}")
        
        val_loss, val_perplexity = evaluate(model, valid_loader, criterion, config.DEVICE)
        print(f"  > Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")
        
        # Save the model if it has the best perplexity so far
        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  > New best model saved to {config.MODEL_SAVE_PATH} (Perplexity: {val_perplexity:.4f})")
            
    print("--- Training Finished ---")
    print(f"Best validation perplexity: {best_val_perplexity:.4f}")
