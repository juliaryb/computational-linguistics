"""
Contains the LSTM model architecture.
"""
import torch
import torch.nn as nn

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
