from torch.utils.data import Dataset
import torch

class LanguageModelDataset(Dataset):
    def __init__(self, text_path, tokenizer, seq_len: int, debug_max_lines=None, stride=None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.stride = stride or seq_len  # non-overlapping by default

        with open(text_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if debug_max_lines is not None:
            lines = lines[:debug_max_lines]
        text = "\n".join(lines)

        ids = tokenizer.encode(text, add_bos_eos=True)
        self.samples = []
        # Build chunks of (seq_len + 1) for strict next-token targets without pad
        for start in range(0, max(0, len(ids) - (seq_len + 1)), self.stride):
            chunk = ids[start: start + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # (T)
        y = torch.tensor(chunk[1:],  dtype=torch.long)  # (T)
        return x, y