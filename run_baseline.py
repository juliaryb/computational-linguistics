import torch
import csv
import time
import math
import os

import config
from train import train_epoch, evaluate
from model import TransformerDecoderOnly
from tokenizer import ensure_spm_tokenizer
from data import LanguageModelDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import random

from benchmark_utils import profile_one_training_step, find_max_batch_size


def run_one_technique(technique, run_tag, baseline_batch_size=None):
    device = config.DEVICE
    print(f"\n=== Running technique: {technique['name']} ===")
    
    # 1. Set the global flag
    config.USE_BF16 = technique["amp_bf16"]
    config.USE_FLASH = technique["flash"]
    config.WINDOW_SIZE = technique["window"]
    config.USE_CKPT = technique["ckpt"]

    # --- Tokenizer ---
    tokenizer = ensure_spm_tokenizer(
        config.TRAIN_PATH,
        config.TOKENIZER_FILE,
        config.VOCAB_SIZE,
    )
    vocab_size = tokenizer.get_vocab_size()

    # --- Data ---
    train_ds = LanguageModelDataset(
        config.TRAIN_PATH, tokenizer, config.SEQ_LEN, None
    )
    val_ds = LanguageModelDataset(
        config.VALID_PATH, tokenizer, config.SEQ_LEN, None
    )

    print(f"Number of training sequences: {len(train_ds)}")
    print(f"Number of validation sequences: {len(val_ds)}")

    # --- setting random seed ---
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # --- Model ---
    def make_model():
        return TransformerDecoderOnly(
            vocab_size=vocab_size,
            d_model=config.TX_D_MODEL,
            n_layer=config.TX_N_LAYER,
            n_head=config.TX_N_HEAD,
            d_ff=config.TX_D_FF,
            dropout=config.TX_DROPOUT,
            pad_id=tokenizer.pad_id,
        )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    if baseline_batch_size is None:
        print("Searching for max batch size...")
        max_batch_size = find_max_batch_size(
            make_model_fn=make_model,
            dataset=train_ds,
            criterion=criterion,
            device=device,
            train_step_fn=profile_one_training_step,
        )
        print(f"Max batch size: {max_batch_size}")
    else:
        max_batch_size = baseline_batch_size
        print(f"Using baseline batch size: {max_batch_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=max_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        generator=g,
        worker_init_fn=seed_worker,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=max_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        generator=g,
        worker_init_fn=seed_worker,
    )

    model = make_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- STEP TIME + MEMORY (20 steps) ---
    step_times = []
    mem_forwards = []
    mem_backwards = []
    peak_mems = []

    train_iter = iter(train_loader)
    for _ in range(20):
        batch = next(train_iter)
        step_t, mem_fwd, mem_bwd, peak_m = profile_one_training_step(
            model, optimizer, criterion, batch, device
        )
        step_times.append(step_t)
        mem_forwards.append(mem_fwd)
        mem_backwards.append(mem_bwd)
        peak_mems.append(peak_m)

    mean_step_time = sum(step_times) / len(step_times)
    mem_forward = sum(mem_forwards) / len(mem_forwards)
    mem_backward = sum(mem_backwards) / len(mem_backwards)
    peak_memory = max(peak_mems)

    # --- FULL EPOCH ---
    train_loss, epoch_time = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_loss, val_ppl = evaluate(
        model, val_loader, criterion, device
    )

    # --- LOG ---
    out_csv = os.path.join(
    config.LOGS_DIR,
    f"benchmark_{technique['name']}_{run_tag}.csv"
    )
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "max_batch_size",
            "mem_forward_mb",
            "mem_backward_mb",
            "peak_mem_mb",
            "mean_step_time_sec",
            "epoch_time_sec",
            "val_perplexity",
            "train_loss",
            "val_loss",
        ])
        writer.writerow([
            max_batch_size,
            mem_forward,
            mem_backward,
            peak_memory,
            mean_step_time,
            epoch_time,
            val_ppl,
            train_loss,
            val_loss,
        ])

    print(f"{technique['name']} done.")
    print(f"Mean step time: {mean_step_time:.4f}s")
    print(f"Peak memory: {peak_memory:.1f} MB")
    print(f"Val perplexity: {val_ppl:.2f}")

    return max_batch_size


if __name__ == "__main__":
    
    TECHNIQUES = [
        {"name": "fp32", "amp_bf16": False, "flash": False, "window": None, "ckpt": False},
        {"name": "bf16", "amp_bf16": True,  "flash": False, "window": None, "ckpt": False},
        # {"name": "fa2",  "amp_bf16": True,  "flash": True,  "window": None, "ckpt": False},
        # {"name": "fa2_window", "amp_bf16": True, "flash": True, "window": 128, "ckpt": False},
        # {"name": "bf16_ckpt", "amp_bf16": True, "flash": False, "window": None, "ckpt": True},
    ]
    
    fp32_max_batch_size = None
    
    for tech in TECHNIQUES:
        if tech["name"] == "fp32":
            # FP32: only max-BS run
            fp32_max_batch_size = run_one_technique(
                technique=tech,
                run_tag="max",
                baseline_batch_size=None,
            )

        else:
            # 1) same batch size as FP32
            run_one_technique(
                technique=tech,
                run_tag="same_bs",
                baseline_batch_size=fp32_max_batch_size,
            )

            # 2) max batch size for this technique
            run_one_technique(
                technique=tech,
                run_tag="max",
                baseline_batch_size=None,
            )
