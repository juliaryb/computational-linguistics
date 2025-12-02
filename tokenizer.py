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
import re
from collections import Counter
from transformers import AutoTokenizer
import config

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE) # one or more “word characters” OR a single character that is not a word or a whitespace


class PretrainedHFTokenizer: # wrapper for a Hugging Face pre-trained tokenizer
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad/bos/eos tokens exist; add them if missing
        special_tokens = {}
        if self.tok.pad_token is None:
            special_tokens["pad_token"] = "<PAD>"
        if self.tok.bos_token is None:
            special_tokens["bos_token"] = "<BOS>"
        if self.tok.eos_token is None:
            special_tokens["eos_token"] = "<EOS>"

        if special_tokens:
            self.tok.add_special_tokens(special_tokens)

        self.pad_id = self.tok.pad_token_id
        self.bos_id = self.tok.bos_token_id
        self.eos_id = self.tok.eos_token_id

        # most HF tokenizers have an unk token; fall back safely if not
        self.unk_id = self.tok.unk_token_id if self.tok.unk_token_id is not None else self.tok.pad_token_id

    def encode(self, text: str, add_bos_eos: bool = True):
        return self.tok.encode(text, add_special_tokens=add_bos_eos) # uses bos/eos as configured above

    def decode(self, token_ids):
        return self.tok.decode(token_ids, skip_special_tokens=True) # removes pad/bos/eos etc from the output

    def get_vocab_size(self) -> int:
        return len(self.tok)


class WhitespaceTokenizer:
    """
    Whitespace + punctuation tokenizer with a fixed-size word-level vocab.
    - splits on whitespace, keeps punctuation as separate tokens
    - uses top-N most frequent tokens from the training corpus
    - everything else goes to <UNK>
    - shares special IDs with prev tokenizers for consitency:
      pad=0, unk=1, bos=2, eos=3
    """
    def __init__(
        self,
        data_path: Optional[str] = None,
        save_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
    ):
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3
        self.idx_to_token: List[str] = []
        self.token_to_idx: Dict[str, int] = {}
        self.vocab_size = 0

        json_path = f"{save_path}.whitespace.json" if save_path else None

        if json_path and os.path.exists(json_path):
            self.load(json_path)
        elif data_path and save_path and vocab_size is not None:
            self.train(data_path, json_path, vocab_size)
        else:
            # Minimal bootstrap vocab
            self.idx_to_token = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
            self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}
            self.vocab_size = len(self.idx_to_token)

    def _tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text)

    def train(self, file_path: str, save_path: str, vocab_size: int):
        """
        Build a word-level vocab:
        - special tokens: 4
        - top (vocab_size - 4) most frequent tokens from the corpus
        """
        counter = Counter()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                toks = self._tokenize(line)
                counter.update(toks)

        # reserve first 4 for specials
        most_common = [tok for tok, _ in counter.most_common(max(vocab_size - 4, 0))]

        self.idx_to_token = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + most_common
        self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}
        self.vocab_size = len(self.idx_to_token)

        if save_path:
            self.save(save_path)

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"idx_to_token": self.idx_to_token}, f, ensure_ascii=False, indent=2)

    def load(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.idx_to_token = data["idx_to_token"]
        self.token_to_idx = {t: i for i, t in enumerate(self.idx_to_token)}
        self.vocab_size = len(self.idx_to_token)

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        toks = self._tokenize(text)
        ids = [self.token_to_idx.get(t, self.unk_id) for t in toks]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        out_tokens: List[str] = []

        for i in token_ids:
            if not (0 <= i < len(self.idx_to_token)):
                continue
            if i in specials:
                continue
            if i == self.unk_id:
                out_tokens.append("<UNK>")
            else:
                out_tokens.append(self.idx_to_token[i])

        # Simple detokeniser: space before words, no space before punctuation
        text = ""
        punct = set(",.!?;:-–—…\"'()[]{}„”")
        for tok in out_tokens:
            if not text:
                text = tok
            elif tok in punct:
                text += tok
            else:
                text += " " + tok
        return text

    def get_vocab_size(self) -> int:
        return self.vocab_size
    

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


def ensure_pretrained_tokenizer() -> PretrainedHFTokenizer:
    return PretrainedHFTokenizer(config.PRETRAINED_MODEL_NAME)

def ensure_whitespace_tokenizer(train_txt: str, save_prefix: str, vocab_size: int) -> WhitespaceTokenizer:
    json_path = f"{save_prefix}.whitespace.json"
    if not os.path.exists(json_path):
        _ = WhitespaceTokenizer(data_path=train_txt, save_path=save_prefix, vocab_size=vocab_size)
    return WhitespaceTokenizer(save_path=save_prefix)

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
