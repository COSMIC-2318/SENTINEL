"""
================================================================
SENTINEL — Module 1 Cross-Modal Attention Pre-training
File: training/pretrain_module1.py

What this file does:
    Uses pre-computed CLIP embeddings from NewsCLIPpings to
    train ONLY the cross-modal attention layer in Module 1.

    We skip downloading raw images entirely. Instead:
    - CLIP already computed 512-dim vectors for every image
    - CLIP already computed 512-dim vectors for every caption
    - We feed these pairs into cross-modal attention
    - The attention layer learns: do these two vectors belong together?

Why this works without raw images:
    CLIP's embeddings already capture semantic meaning.
    A photo of a 2017 protest has a different 512-dim vector
    than a photo of a 2024 protest — even if they look similar.
    The attention layer learns to detect this mismatch.

Output:
    Saves pre-trained cross-modal attention weights to:
    models/cross_modal_pretrained/cross_modal_attention.pt

Usage:
    cd ~/Ankit/SENTINEL
    conda activate sentinel_env
    python training/pretrain_module1.py
================================================================
"""

import os
import sys
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report

# ── Add modules to path ───────────────────────────────────────
SENTINEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SENTINEL_ROOT / "modules" / "module1"))

# ── Paths ─────────────────────────────────────────────────────
NC_PATH    = SENTINEL_ROOT / "data" / "newsclippings"
MODEL_SAVE = SENTINEL_ROOT / "models" / "cross_modal_pretrained"
MODEL_SAVE.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────
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
CONFIG = {
    "embed_dim"    : 512,
    # CLIP ViT-B/32 outputs 512-dim vectors
    # This must match the embedding files

    "num_heads"    : 8,
    # Cross-modal attention uses 8 attention heads
    # Each head learns a different type of image-text relationship
    # 8 heads × 64 dims each = 512 total (matches embed_dim)

    "dropout"      : 0.1,
    # Randomly zeros 10% of neurons during training
    # Forces the model to not rely on any single neuron
    # Another overfitting prevention technique

    "batch_size"   : 64,
    # Larger batch is fine here — we're not loading images
    # Just small numpy vectors, very memory efficient

    "learning_rate": 1e-4,
    # Higher than NLI fine-tuning because we're training
    # from scratch (random weights), not fine-tuning

    "epochs"       : 5,
    # 5 epochs is enough for this focused task

    "seed"         : 42,
}


# ================================================================
# CROSS-MODAL ATTENTION MODULE
# ================================================================
class CrossModalAttention(torch.nn.Module):
    """
    The core of Module 1 — takes text and image embeddings
    and learns to detect when they don't match.

    How it works:
        Text embedding  → Query (what the text is "asking about")
        Image embedding → Key + Value (what the image "offers")

        Attention score = how well text query matches image keys
        High score = text and image are talking about the same thing
        Low score  = mismatch detected

    This is standard transformer cross-attention, but here:
        Query comes from TEXT
        Key + Value come from IMAGE
        Output = text representation enriched with image context
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()

        # Multi-head cross attention
        # embed_dim must be divisible by num_heads
        # 512 / 8 = 64 ✅
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True  # [batch, seq, dim] format
        )

        # Layer normalization — stabilizes training
        # Normalizes each embedding to have mean=0, std=1
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

        # Feed-forward network — processes the attended representation
        # Expands to 4x dimension then back
        # This is standard transformer architecture
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 4),
            torch.nn.GELU(),  # smoother than ReLU for transformers
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim * 4, embed_dim),
            torch.nn.Dropout(dropout),
        )

        # Final classification head
        # Takes the fused representation → outputs fake/real probability
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 1),
            # No sigmoid here — we use BCEWithLogitsLoss
            # which applies sigmoid internally (more numerically stable)
        )

    def forward(self, text_emb: torch.Tensor,
                image_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_emb  : [batch, 512] — CLIP text embeddings
            image_emb : [batch, 512] — CLIP image embeddings

        Returns:
            logits    : [batch, 1] — raw fake/real scores (before sigmoid)
        """
        # Reshape to [batch, 1, 512] for attention
        # MultiheadAttention expects [batch, seq_len, embed_dim]
        # Here seq_len=1 (single vector per modality)
        text_q  = text_emb.unsqueeze(1)   # [B, 1, 512]
        image_k = image_emb.unsqueeze(1)  # [B, 1, 512]
        image_v = image_emb.unsqueeze(1)  # [B, 1, 512]

        # Cross attention: text queries the image
        attended, attention_weights = self.cross_attention(
            query = text_q,
            key   = image_k,
            value = image_v
        )
        # attended: [B, 1, 512]

        # Residual connection + normalization
        # Residual = add original input back to attended output
        # Why? Prevents the original text meaning from being
        # completely overwritten by the image context
        attended = self.norm1(attended + text_q)

        # Feed-forward
        ffn_out  = self.ffn(attended)
        ffn_out  = self.norm2(ffn_out + attended)

        # Squeeze back to [B, 512]
        fused = ffn_out.squeeze(1)

        # Concatenate fused text-image with original image
        # [B, 512] + [B, 512] → [B, 1024]
        # Why concatenate? Gives classifier both:
        # - How text changed after seeing image (fused)
        # - The raw image representation (image_emb)
        combined = torch.cat([fused, image_emb], dim=1)  # [B, 1024]

        # Classify
        logits = self.classifier(combined)  # [B, 1]
        return logits


# ================================================================
# DATASET
# ================================================================
class NewsCLIPpingsDataset(Dataset):
    """
    Loads pre-computed CLIP embeddings and mismatch labels.

    For each annotation:
        text_emb  = text_embeddings[annotation['id']]
        image_emb = image_embeddings[annotation['image_id']]
        label     = 1 if annotation['falsified'] else 0

    When id == image_id: original pair, label=0 (real)
    When id != image_id: image swapped, label=1 (fake/mismatch)
    """

    def __init__(self, split: str):
        """
        Args:
            split: 'train', 'val', or 'test'
        """
        print(f"  Loading {split} split...")

        # Load embeddings
        img_path  = NC_PATH / f"clip_image_embeddings_{split}.pkl"
        text_path = NC_PATH / f"clip_text_embeddings_{split}.pkl"
        json_path = NC_PATH / f"{split}.json"

        print(f"    Loading image embeddings...")
        with open(img_path, "rb") as f:
            self.img_embs = pickle.load(f)   # {id: np.array(512,)}

        print(f"    Loading text embeddings...")
        with open(text_path, "rb") as f:
            self.txt_embs = pickle.load(f)   # {id: np.array(512,)}

        print(f"    Loading annotations...")
        with open(json_path) as f:
            data = json.load(f)

        annotations = data["annotations"]

        # Filter to only annotations where both embeddings exist
        self.samples = []
        skipped = 0
        for ann in annotations:
            text_id  = ann["id"]
            image_id = ann["image_id"]

            if text_id not in self.txt_embs:
                skipped += 1
                continue
            if image_id not in self.img_embs:
                skipped += 1
                continue

            self.samples.append({
                "text_id" : text_id,
                "image_id": image_id,
                "label"   : 1 if ann["falsified"] else 0
            })

        fake_count = sum(1 for s in self.samples if s["label"] == 1)
        real_count = sum(1 for s in self.samples if s["label"] == 0)
        print(f"    ✅ {len(self.samples):,} samples "
              f"(fake: {fake_count:,}, real: {real_count:,}, "
              f"skipped: {skipped:,})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        text_emb  = torch.tensor(
            self.txt_embs[sample["text_id"]], dtype=torch.float32
        )
        image_emb = torch.tensor(
            self.img_embs[sample["image_id"]], dtype=torch.float32
        )
        label = torch.tensor(sample["label"], dtype=torch.float32)

        return {
            "text_emb" : text_emb,   # [512]
            "image_emb": image_emb,  # [512]
            "label"    : label       # scalar
        }


# ================================================================
# EVALUATION
# ================================================================
def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            text_emb  = batch["text_emb"].to(device)
            image_emb = batch["image_emb"].to(device)
            labels    = batch["label"].to(device)

            logits = model(text_emb, image_emb).squeeze(1)
            loss   = loss_fn(logits, labels)
            total_loss += loss.item()

            # Convert logits to binary predictions
            # sigmoid(logit) > 0.5 → predict fake (1)
            preds = (torch.sigmoid(logits) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.long().cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, all_preds, all_labels


# ================================================================
# TRAINING
# ================================================================
def train():
    print("=" * 60)
    print("SENTINEL — Module 1 Cross-Modal Attention Pre-training")
    print("=" * 60)

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ── Load datasets ─────────────────────────────────────────
    print("\n  Loading NewsCLIPpings embeddings...")
    train_dataset = NewsCLIPpingsDataset("train")
    val_dataset   = NewsCLIPpingsDataset("val")
    test_dataset  = NewsCLIPpingsDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size = CONFIG["batch_size"],
        shuffle    = True,
        num_workers= 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = CONFIG["batch_size"],
        shuffle    = False,
        num_workers= 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = CONFIG["batch_size"],
        shuffle    = False,
        num_workers= 0
    )

    # ── Build model ───────────────────────────────────────────
    print("\n  Building Cross-Modal Attention model...")
    model = CrossModalAttention(
        embed_dim = CONFIG["embed_dim"],
        num_heads = CONFIG["num_heads"],
        dropout   = CONFIG["dropout"]
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ── Loss function ─────────────────────────────────────────
    # BCEWithLogitsLoss = Binary Cross Entropy + Sigmoid combined
    # Used for binary classification (fake vs real)
    # pos_weight balances fake/real if dataset is imbalanced
    fake_count = sum(1 for s in train_dataset.samples if s["label"]==1)
    real_count = sum(1 for s in train_dataset.samples if s["label"]==0)
    pos_weight = torch.tensor([real_count / fake_count]).to(DEVICE)
    print(f"  Class balance — fake: {fake_count:,}, real: {real_count:,}")
    print(f"  pos_weight: {pos_weight.item():.3f}")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer + Scheduler ─────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = CONFIG["learning_rate"],
        weight_decay = 0.01
    )

    # CosineAnnealingLR — smoothly decreases learning rate
    # following a cosine curve from max to near 0
    # Better than linear decay for training from scratch
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max = CONFIG["epochs"]
    )

    # ── Training loop ─────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch  = 0

    print(f"\n  Starting training for {CONFIG['epochs']} epochs...")
    print(f"  Batch size : {CONFIG['batch_size']}")
    print(f"  Steps/epoch: {len(train_loader):,}")
    print()

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        correct    = 0
        total      = 0

        for step, batch in enumerate(train_loader):
            text_emb  = batch["text_emb"].to(DEVICE)
            image_emb = batch["image_emb"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            # Forward
            logits = model(text_emb, image_emb).squeeze(1)
            loss   = loss_fn(logits, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds       = (torch.sigmoid(logits) > 0.5).long()
            correct    += (preds == labels.long()).sum().item()
            total      += labels.size(0)

            if (step + 1) % 100 == 0:
                acc = correct / total * 100
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} | Acc: {acc:.1f}%")

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = correct / total * 100

        val_loss, val_f1, _, _ = evaluate(model, val_loader,
                                           loss_fn, DEVICE)

        print(f"\n  ── Epoch {epoch} Summary ──")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Train Acc  : {train_acc:.1f}%")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Val F1     : {val_f1:.4f}")
        print(f"  LR         : {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(
                model.state_dict(),
                MODEL_SAVE / "cross_modal_attention.pt"
            )
            # Save config alongside weights
            torch.save(CONFIG, MODEL_SAVE / "config.pt")
            print(f"  ✅ New best saved! Val F1: {val_f1:.4f}")
        print()

    # ── Final test evaluation ─────────────────────────────────
    print("=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    model.load_state_dict(
        torch.load(MODEL_SAVE / "cross_modal_attention.pt",
                   map_location=DEVICE)
    )
    test_loss, test_f1, preds, labels_true = evaluate(
        model, test_loader, loss_fn, DEVICE
    )

    print(f"\n  Best epoch : {best_epoch}")
    print(f"  Test Loss  : {test_loss:.4f}")
    print(f"  Test F1    : {test_f1:.4f}")
    print()
    print(classification_report(
        labels_true, preds,
        target_names=["Real (matched)", "Fake (mismatched)"]
    ))

    print(f"\n  ✅ Cross-modal attention weights saved to: {MODEL_SAVE}")
    print(f"  Next step: python training/train.py")
    print("=" * 60)


if __name__ == "__main__":
    train()