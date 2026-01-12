import os
import torch
from speakleash import Speakleash
from transformers import AutoTokenizer


def download_high_quality(dataset_name: str, out_path: str, limit=None):
    """
    Downloads high-quality documents from a SpeakLeash dataset and saves them as one merged .txt file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    base_dir = "."
    speakleash_dir = os.path.join(base_dir, "local_datasets")
    os.makedirs(speakleash_dir, exist_ok=True)

    sl = Speakleash(speakleash_dir)
    ds = sl.get(dataset_name).ext_data

    count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for text, meta in ds:
            if meta.get("quality", "").upper() == "HIGH":
                out_f.write(text.strip() + "\n")
                count += 1
                if limit and count >= limit:
                    break

    print(f"Saved {count} high-quality documents from '{dataset_name}' to {out_path}")


# example usage
# download_high_quality("1000_novels_corpus_CLARIN-PL", "data/prose_train.txt")
# download_high_quality("wolne_lektury_corpus", "data/lektury_val.txt")