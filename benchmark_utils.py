import torch
import torch.nn as nn
import math
import time
from config import DEVICE, BATCH_SIZE

def calculate_metrics(model, tokenizer, dataloader, raw_text_path, max_lines=None):
    """
    Calculates metrics on the exact same text subset used by the dataloader.
    Args:
        max_lines (int): Must match the value used in LanguageModelDataset.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction='sum')
    
    total_nll = 0.0
    
    # --- 1. Model Evaluation (Loss Calculation) ---
    use_lstm = hasattr(model, "init_hidden")
    if use_lstm:
        hidden = model.init_hidden(BATCH_SIZE, DEVICE)

    with torch.no_grad():
        for x, y in dataloader:
            if x.shape[0] != BATCH_SIZE: continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            if use_lstm:
                hidden = (hidden[0].detach(), hidden[1].detach())
                logits, hidden = model(x, hidden)
            else:
                logits = model(x)
            
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            total_nll += loss.item()

    # --- 2. Load Raw Text (MATCHING THE SUBSET) ---
    # We must read exactly the same amount of text as the Dataloader did.
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        if max_lines:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines: break
                lines.append(line)
            text = "".join(lines)
        else:
            text = f.read()
    
    # --- 3. Tokenizer Statistics (On the 1MB subset) ---
    words = text.split()
    num_words = len(words)
    num_chars = len(text)
    
    # Encode the 1MB text to get accurate counts
    full_tokens = tokenizer.encode(text, add_bos_eos=False)
    num_tokens_full = len(full_tokens)
    
    # Count OOVs
    unk_count = 0
    if hasattr(tokenizer, 'unk_id'):
        unk_count = full_tokens.count(tokenizer.unk_id)
    
    # Speed Test
    t0 = time.time()
    n_repeats = 5
    for _ in range(n_repeats):
        _ = tokenizer.encode(text, add_bos_eos=False)
    duration = time.time() - t0
    speed_tok_sec = (num_tokens_full * n_repeats) / duration if duration > 0 else 0
    
    # --- 4. Final Metrics ---
    # Now num_tokens_full and num_words come from the same 1MB text.
    return {
        "word_ppl": math.exp(total_nll / num_words) if num_words > 0 else float('inf'),
        "char_ppl": math.exp(total_nll / num_chars) if num_chars > 0 else float('inf'),
        "oov_rate": (unk_count / num_tokens_full) * 100 if num_tokens_full > 0 else 0,
        "oov_count": unk_count,
        "tokens_per_word": num_tokens_full / num_words if num_words > 0 else 0, 
        "tokenizer_speed": speed_tok_sec
    }

def print_qualitative_examples(model, tokenizer, prompt_text):
    """
    Shows how the tokenizer splits text and what the model predicts.
    """
    print(f"--- Qualitative Analysis for {type(tokenizer).__name__} ---")
    
    encoded = tokenizer.encode(prompt_text, add_bos_eos=False)
    decoded_tokens = []
    
    if hasattr(tokenizer, "sp"): 
        decoded_tokens = [tokenizer.sp.id_to_piece(id) for id in encoded]
    elif hasattr(tokenizer, "tok"): 
        decoded_tokens = tokenizer.tok.convert_ids_to_tokens(encoded)
    elif hasattr(tokenizer, "idx_to_token"): 
        decoded_tokens = [tokenizer.idx_to_token[id] if id < len(tokenizer.idx_to_token) else "<UNK>" for id in encoded]
    elif hasattr(tokenizer, "idx_to_char"): 
        decoded_tokens = [tokenizer.idx_to_char[id] for id in encoded]
        
    print(f"Original: {prompt_text}")
    print(f"Tokens:   {decoded_tokens}")
    print(f"IDs:      {encoded}")
    print(f"Count:    {len(encoded)} tokens")
    print("-" * 20)