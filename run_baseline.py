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

from benchmark_utils import profile_one_training_step, find_max_batch_size

def main():
    device = config.DEVICE
    print("Running BASELINE (FP32)")

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

    print("Searching for max batch size (FP32)...")

    max_batch_size = find_max_batch_size(
        make_model_fn=make_model,
        dataset=train_ds,
        criterion=criterion,
        device=device,
        train_step_fn=profile_one_training_step,
    )

    print(f"Max batch size (FP32): {max_batch_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=max_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=max_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
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
    out_csv = os.path.join(config.LOGS_DIR, "baseline_fp32.csv")
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

    print("Baseline done.")
    print(f"Mean step time: {mean_step_time:.4f}s")
    print(f"Peak memory: {peak_memory:.1f} MB")
    print(f"Val perplexity: {val_ppl:.2f}")

if __name__ == "__main__":
    main()
