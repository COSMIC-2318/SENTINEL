"""
================================================================
SENTINEL — Module 2 NLI Fine-tuning on LIAR-PLUS
File: training/finetune_nli.py

What this file does:
    Takes RoBERTa-large (already trained on general NLI)
    and fine-tunes it on LIAR-PLUS political claim-evidence pairs.
    This specializes the model for fake news language patterns.

Why this runs before joint training:
    Joint training needs Module 2 to produce meaningful NLI
    signals. If NLI is random/general, the evidence scores
    fed into Module 3 are noise — not signal.

Output:
    Saves fine-tuned NLI model to:
    models/nli_finetuned/

Usage:
    cd ~/Ankit/SENTINEL
    conda activate sentinel_env
    python training/finetune_nli.py
================================================================
"""

import os
import csv
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report

# ── Paths ─────────────────────────────────────────────────────
SENTINEL_ROOT = Path(__file__).resolve().parent.parent
LIAR_PATH     = SENTINEL_ROOT / "data" / "liar_plus"
MODEL_SAVE    = SENTINEL_ROOT / "models" / "nli_finetuned"
MODEL_SAVE.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────
# On Mac M4, use "mps" for GPU acceleration
# mps = Metal Performance Shaders (Apple's GPU framework)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using MPS (Apple M4 GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  Using CPU — training will be slow")

# ── Hyperparameters ───────────────────────────────────────────
# These control how training behaves
# Explained below each one
CONFIG = {
    "model_name"   : "cross-encoder/nli-roberta-base",
    # Why this model?
    # It's RoBERTa already fine-tuned on NLI (SNLI + MultiNLI)
    # We fine-tune it further on LIAR-PLUS
    # This is faster than starting from raw RoBERTa

    "max_length"   : 256,
    # Maximum tokens per input (claim + evidence combined)
    # 256 is enough for most claim-evidence pairs
    # RoBERTa max is 512 but 256 trains 2x faster

    "batch_size"   : 16,
    # 16 samples per gradient update
    # Safe for M4 Mac mini memory

    "learning_rate": 2e-5,
    # How big each weight update step is
    # 2e-5 is standard for fine-tuning transformers
    # Too high → model forgets what it learned (catastrophic forgetting)
    # Too low → training is too slow

    "epochs"       : 4,
    # How many times we go through the full dataset
    # 3 epochs is standard for NLI fine-tuning
    # More epochs → overfitting risk

    "warmup_ratio" : 0.1,
    # For the first 10% of training steps, learning rate
    # gradually increases from 0 to full value
    # Why? Sudden large updates at the start can destabilize
    # a pre-trained model

    "freeze_layers": 8,
    # Freeze the first 8 transformer layers
    # Why? Lower layers = general language understanding
    # We want to keep that and only adapt the top layers
    # to fake news domain

    "train_ratio"  : 0.8,
    "val_ratio"    : 0.1,
    "test_ratio"   : 0.1,
    "seed"         : 42,
}

# ── Label mapping ─────────────────────────────────────────────
# LIAR-PLUS has 6 labels → we map to 3 NLI labels
# 0 = ENTAILMENT   (evidence supports claim)
# 1 = NEUTRAL      (evidence unrelated)
# 2 = CONTRADICTION (evidence contradicts claim)

LIAR_TO_NLI = {
    "true"       : 0,  # ENTAILMENT
    "mostly-true": 0,  # ENTAILMENT
    "half-true"  : 1,  # NEUTRAL
    "barely-true": 2,  # CONTRADICTION
    "false"      : 2,  # CONTRADICTION
    "pants-fire" : 2,  # CONTRADICTION
}

LABEL_NAMES = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}


# ================================================================
# DATASET
# ================================================================
class LiarPlusDataset(Dataset):
    """
    Loads LIAR-PLUS TSV files and formats them for NLI fine-tuning.

    LIAR-PLUS TSV columns (0-indexed):
    0  = id
    1  = label (true/false/etc)
    2  = statement (the claim)
    3  = subject
    4  = speaker
    5  = speaker job
    6  = state
    7  = party
    8  = barely_true_count
    9  = false_count
    10 = half_true_count
    11 = mostly_true_count
    12 = pants_fire_count
    13 = context
    14 = justification  ← this is our "evidence"

    We use:
    - Column 2 (statement) as the HYPOTHESIS (the claim to verify)
    - Column 14 (justification) as the PREMISE (the evidence)
    - Column 1 (label) mapped to NLI label (0, 1, 2)
    """

    def __init__(self, filepath: str, tokenizer, max_length: int = 256):
        self.samples    = []
        self.tokenizer  = tokenizer
        self.max_length = max_length

        skipped = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # Skip malformed rows
                if len(row) < 16:
                    skipped += 1
                    continue

                label_str     = row[2].strip().lower()
                statement     = row[3].strip()   # the claim
                justification = row[15].strip()  # the evidence

                # Skip if label not recognized
                if label_str not in LIAR_TO_NLI:
                    skipped += 1
                    continue

                # Skip if either text is too short
                if len(statement) < 10 or len(justification) < 10:
                    skipped += 1
                    continue

                nli_label = LIAR_TO_NLI[label_str]

                self.samples.append({
                    "premise"   : justification,  # evidence
                    "hypothesis": statement,       # claim
                    "label"     : nli_label
                })

        print(f"  Loaded {len(self.samples)} samples "
              f"(skipped {skipped} malformed/short)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize premise + hypothesis together
        # The tokenizer formats them as:
        # [CLS] premise [SEP] hypothesis [SEP]
        # This is exactly what NLI models expect
        encoding = self.tokenizer(
            sample["premise"],
            sample["hypothesis"],
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt"
        )

        return {
            "input_ids"     : encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label"         : torch.tensor(sample["label"],
                                           dtype=torch.long)
        }


# ================================================================
# FREEZE LAYERS
# ================================================================
def freeze_bottom_layers(model, num_layers_to_freeze: int):
    """
    Freezes the bottom N transformer layers.

    Why freeze lower layers?
        Lower layers of RoBERTa learn general things like
        grammar, word meanings, sentence structure.
        We want to KEEP this knowledge.

        Upper layers learn task-specific patterns.
        We want to ADAPT these to fake news language.

    Freezing = setting requires_grad=False
    This means PyTorch skips computing gradients for these
    parameters — they don't get updated during training.
    """
    # Freeze embedding layer
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    # Freeze bottom N encoder layers
    for i in range(num_layers_to_freeze):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False

    # Count frozen vs trainable
    frozen    = sum(p.numel() for p in model.parameters()
                    if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    total     = frozen + trainable

    print(f"  Frozen parameters    : {frozen:,} "
          f"({frozen/total*100:.1f}%)")
    print(f"  Trainable parameters : {trainable:,} "
          f"({trainable/total*100:.1f}%)")


# ================================================================
# EVALUATION
# ================================================================
def evaluate(model, loader, device):
    """
    Runs model on a DataLoader and returns loss + F1 score.

    Why F1 and not accuracy?
        Our 3 classes may not be perfectly balanced.
        F1 macro averages precision+recall across all classes
        equally — more honest than accuracy.
    """
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            loss   = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, f1, all_preds, all_labels


# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================
def train():
    print("=" * 60)
    print("SENTINEL — NLI Fine-tuning on LIAR-PLUS")
    print("=" * 60)

    # ── Load tokenizer + model ────────────────────────────────
    print(f"\n  Loading model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model     = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=3  # ENTAILMENT, NEUTRAL, CONTRADICTION
    )

    # ── Freeze bottom layers ──────────────────────────────────
    print(f"\n  Freezing bottom {CONFIG['freeze_layers']} layers...")
    freeze_bottom_layers(model, CONFIG["freeze_layers"])
    model = model.to(DEVICE)

    # ── Load datasets ─────────────────────────────────────────
    print(f"\n  Loading LIAR-PLUS from {LIAR_PATH}")
    train_file = LIAR_PATH / "train.tsv"
    val_file   = LIAR_PATH / "val.tsv"
    test_file  = LIAR_PATH / "test.tsv"

    train_dataset = LiarPlusDataset(str(train_file), tokenizer,
                                     CONFIG["max_length"])
    val_dataset   = LiarPlusDataset(str(val_file),   tokenizer,
                                     CONFIG["max_length"])
    test_dataset  = LiarPlusDataset(str(test_file),  tokenizer,
                                     CONFIG["max_length"])

    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=False)
    test_loader  = DataLoader(test_dataset,
                              batch_size=CONFIG["batch_size"],
                              shuffle=False)

    print(f"\n  Train   : {len(train_dataset):,} samples")
    print(f"  Val     : {len(val_dataset):,} samples")
    print(f"  Test    : {len(test_dataset):,} samples")

    # ── Label distribution ────────────────────────────────────
    labels = [s["label"] for s in train_dataset.samples]
    for i, name in LABEL_NAMES.items():
        count = labels.count(i)
        print(f"  {name:15}: {count:,} ({count/len(labels)*100:.1f}%)")

    # ── Optimizer ─────────────────────────────────────────────
    # AdamW is Adam with weight decay
    # Weight decay is a regularization technique that prevents
    # weights from growing too large — reduces overfitting
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01
    )

    # ── Learning rate scheduler ───────────────────────────────
    # Warmup: gradually increase LR for first 10% of steps
    # Then linearly decrease LR to 0 by end of training
    # This stabilizes fine-tuning of pre-trained models
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    # Weight NEUTRAL 2x because it has fewest samples
    # This forces model to pay more attention to rare class
    class_weights = torch.tensor([1.0, 2.0, 1.0]).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── Training loop ─────────────────────────────────────────
    best_val_f1  = 0.0
    best_epoch   = 0
    history      = []

    print(f"\n  Starting training for {CONFIG['epochs']} epochs...")
    print(f"  Total steps : {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print()

    for epoch in range(1, CONFIG["epochs"] + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct    = 0
        total      = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            # Compute loss
            loss = loss_fn(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            # If any gradient is larger than 1.0, scale them all down
            # This is especially important for transformers
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            # Print progress every 50 steps
            if (step + 1) % 50 == 0:
                acc = correct / total * 100
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} | Acc: {acc:.1f}%")

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = correct / total * 100

        # ── Validate ──────────────────────────────────────────
        val_loss, val_f1, _, _ = evaluate(model, val_loader, DEVICE)

        print(f"\n  ── Epoch {epoch} Summary ──")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Train Acc  : {train_acc:.1f}%")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Val F1     : {val_f1:.4f}")

        history.append({
            "epoch"     : epoch,
            "train_loss": avg_train_loss,
            "train_acc" : train_acc,
            "val_loss"  : val_loss,
            "val_f1"    : val_f1
        })

        # ── Save best model ───────────────────────────────────
        # We save the model with the highest validation F1
        # NOT the model from the last epoch
        # Why? The last epoch may have overfit
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            model.save_pretrained(str(MODEL_SAVE))
            tokenizer.save_pretrained(str(MODEL_SAVE))
            print(f"  ✅ New best model saved! Val F1: {val_f1:.4f}")
        print()

    # ── Final test evaluation ─────────────────────────────────
    print("=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    # Load the best saved model for final evaluation
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_SAVE)
    ).to(DEVICE)

    test_loss, test_f1, preds, labels_true = evaluate(
        best_model, test_loader, DEVICE
    )

    print(f"\n  Best epoch : {best_epoch}")
    print(f"  Test Loss  : {test_loss:.4f}")
    print(f"  Test F1    : {test_f1:.4f}")
    print()
    print("  Per-class breakdown:")
    print(classification_report(
        labels_true, preds,
        target_names=["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    ))

    # Save training history
    with open(MODEL_SAVE / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ Fine-tuned NLI model saved to: {MODEL_SAVE}")
    print(f"  Next step: python training/train.py")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    train()