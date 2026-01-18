"""
Contains the LSTM model architecture.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import config

try:
    from flash_attn import flash_attn_func
    print("Flash attention successfully imported")
except Exception:
    flash_attn_func = None

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1, tokenizer = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Embedding Layer
        #    Turns token IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_id) 
        
        self.dropout = nn.Dropout(dropout)

        # 2. LSTM Layer
        #    Processes the sequences of embedding vectors.
        #    batch_first=True makes input/output tensors (batch_size, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. Output Layer (Linear Head)
        #    Maps the LSTM's hidden state to scores for each token in the vocab.
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, seq_len)
        
        # 1. Get embeddings
        # embeds shape: (batch_size, seq_len, embed_dim)
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        
        # 2. Pass through LSTM
        #    lstm_out shape: (batch_size, seq_len, hidden_dim)
        #    hidden is a tuple (h_n, c_n), each with shape (num_layers, batch_size, hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # 3. Pass through output layer
        #    logits shape: (batch_size, seq_len, vocab_size)
        logits = self.fc_out(lstm_out)
        
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        # Create two tensors for the hidden state (h_0) and cell state (c_0)
        # Shape: (num_layers, batch_size, hidden_dim)
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        )



# --- Positional Encoding (sinusoidal) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)                        # (T, C)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                      # (1, T, C) for batch_first
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):                                          # x: (B, T, C)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# worked but gave warning
# def causal_mask(T: int, device):
#     # (T, T) with -inf above main diagonal (no looking ahead)
#     m = torch.full((T, T), float("-inf"), device=device)
#     return torch.triu(m, diagonal=1)

def causal_mask(T: int, device):
    # boolean mask: True where attention is NOT allowed (future positions)
    # shape (T, T)
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.head_dim = d_model // n_head
        self.attn_dropout = dropout

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _flash_attention(self, x, key_padding_mask=None):
        # x: (B, T, C)
        if flash_attn_func is None:
            raise RuntimeError("config.USE_FLASH=True but flash-attn is not importable.")

        if key_padding_mask is not None and key_padding_mask.any().item():
            raise RuntimeError(
                "FlashAttention path received key_padding_mask with True values (padding present). "
                "Either ensure fixed-length non-padded batches, or implement varlen FlashAttention."
            )

        B, T, C = x.shape
        H = self.n_head
        D = self.head_dim

        # Reuse MHA parameters (no extra params added)
        w = self.self_attn.in_proj_weight      # (3C, C)
        b = self.self_attn.in_proj_bias        # (3C,)
        qkv = F.linear(x, w, b)                # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)         # each (B, T, C)


        # reshape to (B, T, H, D) as expected by flash_attn_func
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)

        # dropout_p should be 0.0 during evaluation
        dropout_p = self.attn_dropout if self.training else 0.0

        # window_size=(-1, -1) = full attention
        #   (W, 0)   = causal sliding-window (only lookback)
        window_size = (-1, -1)
        if getattr(config, "WINDOW_SIZE", None) is not None:
            ws = config.WINDOW_SIZE
            if ws is not None:
                window_size = (ws, 0)

        attn_out = flash_attn_func(
            q, k, v,
            dropout_p=dropout_p,
            causal=True,                # lower-triangular mask -> "seeing the future"
            window_size=window_size,    # see window_size tokens back, but no future ones
        )  # (B, T, H, D)

        attn_out = attn_out.reshape(B, T, C)   # merge heads
        attn_out = F.linear(
            attn_out, 
            self.self_attn.out_proj.weight, 
            self.self_attn.out_proj.bias)
        return attn_out

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, T, C)
        if getattr(config, "USE_FLASH", False):
            attn_out = self._flash_attention(x, key_padding_mask=key_padding_mask)
        else:
            attn_out, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

        x = self.ln1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.drop(ff_out))
        return x

class TransformerDecoderOnly(nn.Module):
    """
    Simple GPT-like decoder-only LM:
      Embedding + PosEnc + N x (SelfAttn + FFN) + LM head
    """
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int, d_ff: int,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.posenc = PositionalEncoding(d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but helps)
        self.lm_head.weight = self.embed.weight

    def forward(self, x):                         # x: (B, T) token ids
        B, T = x.shape
        h = self.embed(x)                         # (B, T, C)
        h = self.posenc(h)                        # (B, T, C)

        # boolean mask
        attn_mask = causal_mask(T, x.device)

        # key_padding_mask: True for pads to be masked out in attention
        key_padding_mask = (x == self.pad_id)     # (B, T), bool

        for blk in self.blocks:
            # this adds gradient checkpointing
            if getattr(config, "USE_CKPT", False) and self.training:
                # checkpoint requires a function whose *inputs are tensors*
                def custom_forward(h_):
                    return blk(h_, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

                h = checkpoint(custom_forward, h, use_reentrant=False)
            else:
                h = blk(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        h = self.ln_f(h)                          # (B, T, C)
        logits = self.lm_head(h)                  # (B, T, V)
        return logits