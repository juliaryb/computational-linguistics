"""
Script for generating text from a trained model.
"""
import torch
import torch.nn.functional as F
import config
from tokenizer import ensure_char_tokenizer, ensure_spm_tokenizer
from model import LSTMModel, TransformerDecoderOnly
import sys
import os

def generate_text(model, tokenizer, prompt: str, max_len: int, temperature: float = 1.0):
    """
    Generates text given a prompt.
    
    Args:
        model: The trained LSTMModel.
        tokenizer: The tokenizer.
        prompt (str): The starting text.
        max_len (int): Maximum number of tokens to generate.
        temperature (float): Controls randomness. 
                             < 1.0 = more predictable, > 1.0 = more random.
    """
    model.eval() # Set model to evaluation mode
    device = next(model.parameters()).device # Get model's device
    
    # --- Process the prompt ---
    token_ids = tokenizer.encode(prompt)
    
    if len(token_ids) == 0:
        # fallback in case encoding returns empty
        token_ids = [getattr(tokenizer, "bos_id", 1)]  # 1 is a common BOS fallback


    use_lstm = hasattr(model, "init_hidden")
    print(use_lstm)
    if use_lstm:
        # Initialize hidden state
        # We use a batch size of 1 for generation
        hidden = model.init_hidden(1, device)
    
        # Feed the prompt tokens through the model one by one
        # to build up the hidden state
        if len(token_ids) > 1:
            prompt_tensor = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                _, hidden = model(prompt_tensor, hidden)
    
        # Get the last token of the prompt to start generation
        current_token_tensor = torch.tensor([token_ids[-1]], dtype=torch.long).unsqueeze(0).to(device)

        # --- Generate new tokens ---
        generated_ids = list(token_ids)
    
        for _ in range(max_len):
            with torch.no_grad():
                # Get logits for the *last* token
                # current_token_tensor shape: (1, 1)
                logits, hidden = model(current_token_tensor, hidden)
                
                # logits shape: (1, 1, vocab_size) -> (1, vocab_size)
                logits_squeezed = logits.squeeze(1) 
                
                # Apply temperature
                # (Higher temp -> flatter distribution -> more random)
                logits_scaled = logits_squeezed / temperature
                
                # Get probabilities
                probs = F.softmax(logits_scaled, dim=-1)
                
                # Sample from the distribution
                # multinomial(probs, 1) returns shape (1, 1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # Add the new token
                generated_ids.append(next_token_id)
                
                # Prepare the next input (the token we just generated)
                current_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                
    else:
            # For a decoder-only transformer we feed the full context each step
            # and take the logits at the last position.

            generated_ids = list(token_ids)

            for _ in range(max_len):
                with torch.no_grad():
                    # inp: (1, T)
                    inp = torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)
                    # logits: (1, T, vocab_size)
                    logits = model(inp)

                    # Take last position logits: (1, vocab_size)
                    logits_last = logits[:, -1, :]

                    # Apply temperature
                    logits_scaled = logits_last / max(1e-8, temperature)

                    # Probabilities and sample
                    probs = F.softmax(logits_scaled, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(next_token_id)

    # Decode the generated IDs into a string
    return tokenizer.decode(generated_ids)



# how to run: `python generate.py "My prompt"`
if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "Tutaj jakiś prompt"
    
    print(f"--- Loading model from {config.MODEL_SAVE_PATH} ---")
    
    # --- 1. Load Tokenizer ---
    # Construct the full path just like in train.py
    tokenizer_path = os.path.join(config.DATA_DIR, config.TOKENIZER_FILE)
    tokenizer_path_ext = (tokenizer_path + ".json" 
                          if config.TOKENIZER_TYPE == "char" 
                          else tokenizer_path + ".model")

    if not os.path.exists(tokenizer_path_ext):
        print(f"Error: Tokenizer file not found at {tokenizer_path_ext}")
        print("Please run `python train.py` first to create a tokenizer.")
        sys.exit(1)
    
    if config.TOKENIZER_TYPE == "char":
        tokenizer = ensure_char_tokenizer("dummy_filepath", tokenizer_path)
    elif config.TOKENIZER_TYPE == "spm":
        tokenizer = ensure_spm_tokenizer("dummy_filepath", tokenizer_path)
    else:
        raise ValueError(f"Unknown TOKENIZER_TYPE={config.TOKENIZER_TYPE}")
    
    vocab_size = tokenizer.get_vocab_size()
    
    # --- 2. Load Model ---
    if config.MODEL_ARCH == "lstm":
        model = LSTMModel(
            vocab_size=vocab_size,
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            tokenizer=tokenizer
        ).to(config.DEVICE)

    elif config.MODEL_ARCH == "transformer":
        model = TransformerDecoderOnly(
            vocab_size=vocab_size,
            d_model=config.TX_D_MODEL,
            n_layer=config.TX_N_LAYER,
            n_head=config.TX_N_HEAD,
            d_ff=config.TX_D_FF,
            dropout=config.TX_DROPOUT,
            pad_id=tokenizer.pad_id
        ).to(config.DEVICE)
    else:
        raise ValueError(f"Unknown MODEL_ARCH={config.MODEL_ARCH}")

    # load the right checkpoint
    ckpt_path = config.MODEL_SAVE_PATH

    try:
        # Add weights_only=True to fix the FutureWarning
        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        print("Please run `python train.py` first to train and save a model.")
        sys.exit(1)
        
    print(f"Model loaded. Generating text with prompt: '{prompt}'")
    print("---")
    
    # --- 3. Generate ---
    generated_text = generate_text(model, tokenizer, prompt=prompt, max_len=200, temperature=0.8)
    
    print(generated_text)
    print("\n---")

