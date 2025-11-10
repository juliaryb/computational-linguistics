"""
Two tiny tokenizers with the SAME API:
- CharTokenizer (pad=0, unk=1, bos=2, eos=3)
- SPMTokenizer  (pad=3, unk=0, bos=1, eos=2 by training args)

Both include:
  encode(text, add_bos_eos: bool = True) -> List[int]
  decode(ids: List[int]) -> str
  get_vocab_size() -> int
  pad_id, bos_id, eos_id, unk_id  (attributes)
"""

import json
import os
from typing import List, Dict, Optional
import sentencepiece as spm



class CharTokenizer:
    def __init__(self, data_path: Optional[str] = None, save_path: Optional[str] = None):
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3
        self.idx_to_char: List[str] = []
        self.char_to_idx: Dict[str, int] = {}
        self.vocab_size = 0

        json_path = f"{save_path}.json" if save_path else None
        if json_path and os.path.exists(json_path):
            self.load(json_path)
        elif data_path and save_path:
            self.train(data_path, json_path)
        else:
            # minimal bootstrapped vocab
            self.idx_to_char = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
            self.char_to_idx = {ch: i for i, ch in enumerate(self.idx_to_char)}
            self.vocab_size = len(self.idx_to_char)

    def train(self, file_path: str, save_path: str):
        vocab = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                vocab.update(line)
        sorted_vocab = sorted(list(vocab))
        self.idx_to_char = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + sorted_vocab
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.idx_to_char)}
        self.vocab_size = len(self.idx_to_char)
        if save_path:
            self.save(save_path)

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"idx_to_char": self.idx_to_char}, f, ensure_ascii=False, indent=2)

    def load(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.idx_to_char = data["idx_to_char"]
        self.char_to_idx = {ch: i for i, ch in enumerate(self.idx_to_char)}
        self.vocab_size = len(self.idx_to_char)

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        ids = [self.char_to_idx.get(ch, self.unk_id) for ch in text]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        out = []
        for i in token_ids:
            if 0 <= i < len(self.idx_to_char):
                if i in specials:
                    continue
                if i == self.unk_id:
                    out.append("�")
                else:
                    out.append(self.idx_to_char[i])
        return "".join(out)

    def get_vocab_size(self) -> int:
        return self.vocab_size

class SPMTokenizer:
    def __init__(self, data_path: Optional[str] = None, save_path: Optional[str] = None):
        self.sp = spm.SentencePieceProcessor()
        model_path = f"{save_path}.model" if save_path else None

        if model_path and os.path.exists(model_path):
            self.sp.load(model_path)
        elif data_path and save_path:
            # Train a small, sensible model
            args = [
                f"--input={data_path}",
                f"--model_prefix={save_path}",
                "--model_type=unigram",
                "--vocab_size=8000",
                "--character_coverage=1.0",
                "--normalization_rule_name=nfkc",
                "--add_dummy_prefix=true",
                "--byte_fallback=false",
                "--unk_id=0", "--bos_id=1", "--eos_id=2", "--pad_id=3",
                "--input_sentence_size=2000000",
                "--shuffle_input_sentence=true",
            ]
            spm.SentencePieceTrainer.train(" ".join(args))
            self.sp.load(model_path)
        else:
            raise ValueError("SPMTokenizer needs either an existing model (save_path) or data_path+save_path to train.")

        # ids (from the trained model)
        self.unk_id = int(self.sp.unk_id())
        self.bos_id = int(self.sp.bos_id())
        self.eos_id = int(self.sp.eos_id())
        self.pad_id = int(self.sp.pad_id())

    def encode(self, text: str, add_bos_eos: bool = True):
        return self.sp.encode(text, out_type=int,
                              add_bos=add_bos_eos, add_eos=add_bos_eos)

    def decode(self, token_ids):
        return self.sp.decode(token_ids)

    def get_vocab_size(self):
        return int(self.sp.get_piece_size())


def ensure_char_tokenizer(train_txt: str, save_prefix: str) -> CharTokenizer:
    json_path = f"{save_prefix}.json"
    if not os.path.exists(json_path):
        _ = CharTokenizer(data_path=train_txt, save_path=save_prefix)
    return CharTokenizer(save_path=save_prefix)

def ensure_spm_tokenizer(train_txt: str, save_prefix: str) -> SPMTokenizer:
    model_path = f"{save_prefix}.model"
    if not os.path.exists(model_path):
        _ = SPMTokenizer(data_path=train_txt, save_path=save_prefix)
    return SPMTokenizer(save_path=save_prefix)
