"""
================================================================
SENTINEL — RoBERTa Fine-tuning on WELFake
File: training/train_roberta.py

What this does:
    Fine-tunes RoBERTa-base directly on 72K WELFake articles.
    No module integration complexity. Just text → label.

    This gives us real, honest F1 numbers for the demo.
    The trained weights power Module 1's text branch.

Expected results:
    ~2-3 hours on Mac mini M4
    Val F1: 0.92-0.95

Usage:
    cd ~/Ankit/SENTINEL
    conda activate sentinel_env
    python training/train_roberta.py
================================================================
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, classification_report

# ── Paths ─────────────────────────────────────────────────────
SENTINEL_ROOT = Path(__file__).resolve().parent.parent
WELFAKE_CSV   = SENTINEL_ROOT / "data" / "WELFake_Dataset.csv"
SAVE_DIR      = SENTINEL_ROOT / "models" / "roberta_welfake"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using MPS (Apple M4 GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  Using CPU")

# ── Config ────────────────────────────────────────────────────
CONFIG = {
    "model_name"   : "roberta-base",
    "max_length"   : 512,
    "batch_size"   : 16,
    "epochs"       : 3,       # RoBERTa fine-tuning needs only 3 epochs
    "learning_rate": 2e-5,
    "weight_decay" : 0.01,
    "seed"         : 42,
    "val_split"    : 0.1,
    "test_split"   : 0.1,
}


# ================================================================
# DATASET
# ================================================================
class WELFakeTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        title = str(row.get("title", "")) if pd.notna(row.get("title", "")) else ""
        text  = str(row["text"]).strip()
        combined = f"{title} {text}".strip()

        enc = self.tokenizer(
            combined,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt"
        )

        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(int(row["label"]), dtype=torch.long),
        }


# ================================================================
# LOAD & SPLIT DATA
# ================================================================
def load_data(tokenizer):
    print(f"\n  Loading WELFake from {WELFAKE_CSV}...")
    df = pd.read_csv(WELFAKE_CSV)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() >= 30]
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    fake_c = (df["label"] == 1).sum()
    real_c = (df["label"] == 0).sum()
    print(f"  Total: {len(df)} samples (Fake: {fake_c} | Real: {real_c})")

    total      = len(df)
    test_size  = int(total * CONFIG["test_split"])
    val_size   = int(total * CONFIG["val_split"])
    train_size = total - val_size - test_size

    # Shuffle before splitting
    df = df.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)

    train_df = df.iloc[:train_size]
    val_df   = df.iloc[train_size:train_size + val_size]
    test_df  = df.iloc[train_size + val_size:]

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = WELFakeTextDataset(train_df, tokenizer, CONFIG["max_length"])
    val_ds   = WELFakeTextDataset(val_df,   tokenizer, CONFIG["max_length"])
    test_ds  = WELFakeTextDataset(test_df,  tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ================================================================
# EVALUATE
# ================================================================
def evaluate(loader, model):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            out  = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels)
            total_loss += out.loss.item()
            preds = out.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, f1, all_preds, all_labels


# ================================================================
# TRAIN
# ================================================================
def train():
    print("=" * 60)
    print("SENTINEL — RoBERTa Fine-tuning on WELFake")
    print("=" * 60)

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ── Tokenizer + Model ─────────────────────────────────────
    print(f"\n  Loading {CONFIG['model_name']}...")
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG["model_name"])
    model     = RobertaForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=2
    ).to(DEVICE)
    print("  ✅ Model loaded")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = load_data(tokenizer)

    # ── Optimizer + Scheduler ─────────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr=CONFIG["learning_rate"],
                      weight_decay=CONFIG["weight_decay"])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["epochs"]
    )

    best_val_f1 = 0.0
    best_epoch  = 0
    history     = []

    print(f"\n  Training for {CONFIG['epochs']} epochs")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Device: {DEVICE}\n")

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        all_preds  = []
        all_labels = []

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            out  = model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = out.logits.argmax(dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (step + 1) % 100 == 0:
                running_f1 = f1_score(all_labels, all_preds,
                                      average="macro", zero_division=0)
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | F1: {running_f1:.4f}")

        train_f1 = f1_score(all_labels, all_preds,
                             average="macro", zero_division=0)

        val_loss, val_f1, _, _ = evaluate(val_loader, model)

        print(f"\n  ── Epoch {epoch} Summary ──")
        print(f"  Train Loss : {train_loss/len(train_loader):.4f}")
        print(f"  Train F1   : {train_f1:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Val F1     : {val_f1:.4f}")

        history.append({
            "epoch"     : epoch,
            "train_loss": train_loss / len(train_loader),
            "train_f1"  : train_f1,
            "val_loss"  : val_loss,
            "val_f1"    : val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            model.save_pretrained(str(SAVE_DIR))
            tokenizer.save_pretrained(str(SAVE_DIR))
            print(f"  ✅ Best model saved! Val F1: {val_f1:.4f}")
        print()

    # ── Final Test ────────────────────────────────────────────
    print("=" * 60)
    print("  Final Test Evaluation")
    print("=" * 60)

    # Load best model for test evaluation
    best_model = RobertaForSequenceClassification.from_pretrained(
        str(SAVE_DIR)
    ).to(DEVICE)

    test_loss, test_f1, preds, labels_true = evaluate(test_loader, best_model)

    print(f"\n  Best epoch : {best_epoch}")
    print(f"  Test Loss  : {test_loss:.4f}")
    print(f"  Test F1    : {test_f1:.4f}")
    print()
    print(classification_report(
        labels_true, preds, target_names=["Real", "Fake"]
    ))

    with open(SAVE_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ Model saved to: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()