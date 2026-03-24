"""
================================================================
SENTINEL — Joint End-to-End Training
File: training/train.py

Trains Modules 1, 2, 3 together on FakeNewsNet PolitiFact.
Module 4 excluded — LLaMA-3 has 8B params, no gradient support.

Output: models/sentinel_joint/best_model.pt

Usage:
    cd ~/Ankit/SENTINEL
    conda activate sentinel_env
    python training/train.py
================================================================
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────
SENTINEL_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(SENTINEL_ROOT / "modules" / "module1_multimodal"))
sys.path.insert(0, str(SENTINEL_ROOT / "modules" / "module2_rag"))
sys.path.insert(0, str(SENTINEL_ROOT / "modules" / "module3_gnn"))
sys.path.insert(0, str(SENTINEL_ROOT / "training"))

from module1 import Module1
from module3 import run_module3
from dataset import create_data_splits

DATA_ROOT      = (SENTINEL_ROOT / "data" / "fakenewsnet" /
                  "FakeNewsNet" / "code" / "fakenewsnet_dataset")
NLI_MODEL_PATH = SENTINEL_ROOT / "models" / "nli_finetuned"
SAVE_PATH      = SENTINEL_ROOT / "models" / "sentinel_joint"
SAVE_PATH.mkdir(parents=True, exist_ok=True)

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

# ── Hyperparameters ───────────────────────────────────────────
CONFIG = {
    "batch_size"   : 8,
    "learning_rate": 1e-4,
    "epochs"       : 10,
    "dropout"      : 0.3,
    "seed"         : 42,
}


# ================================================================
# CLASSIFICATION HEAD
# ================================================================
class SentinelClassifier(torch.nn.Module):
    """
    Combines scalar outputs from all 3 modules.

    Input: 9-dim vector
        Module 1 fake_prob   [1]
        Module 1 mismatch    [1]
        Module 2 evidence    [6]
        Module 3 fake_prob   [1]
    Output: single logit → sigmoid → P(Fake)
    """
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


# ================================================================
# MODULE 2 — EVIDENCE SCORES
# ================================================================
def get_evidence_scores(texts, nli_model, nli_tokenizer, device):
    """6-dim NLI evidence score per article."""
    scores = torch.zeros(len(texts), 6).to(device)
    nli_model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            mid        = len(text) // 2
            premise    = text[:mid][:512]
            hypothesis = text[mid:][:512]
            if len(premise) < 10 or len(hypothesis) < 10:
                continue
            enc = nli_tokenizer(
                premise, hypothesis,
                max_length=256, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            out   = nli_model(
                input_ids      = enc["input_ids"].to(device),
                attention_mask = enc["attention_mask"].to(device)
            )
            probs = torch.softmax(out.logits, dim=1)[0]
            scores[i, 0] = probs[0]
            scores[i, 1] = probs[1]
            scores[i, 2] = probs[2]
            scores[i, 3] = probs.max()
            scores[i, 4] = probs.min()
            scores[i, 5] = probs.max() - probs.min()
    return scores


# ================================================================
# FORWARD PASS
# ================================================================
def run_step(batch, module1, nli_model, nli_tokenizer,
             classifier, loss_fn, device):

    texts  = batch["text"]
    images = batch["image"]
    labels = batch["label"].to(device)

    # ── Module 1 ──────────────────────────────────────────────
    m1_fake_probs = []
    m1_mismatches = []
    m1_fusions    = []

    for i in range(len(texts)):
        # Denormalise tensor → PIL image
        img_t   = images[i].permute(1, 2, 0).numpy()
        img_t   = ((img_t * [0.26862954, 0.26130258, 0.27577711] +
                    [0.48145466, 0.4578275,  0.40821073]) * 255)
        img_t   = np.clip(img_t, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_t)

        try:
            result    = module1.predict(texts[i], pil_img)
            fake_prob = float(result.get("fake_probability", 0.5))
            mismatch  = float(result.get("attention_score", 0.5))
            fusion    = result.get("fusion_vector", None)
        except Exception:
            fake_prob = 0.5
            mismatch  = 0.5
            fusion    = None

        m1_fake_probs.append(fake_prob)
        m1_mismatches.append(mismatch)

        # ── FIX: .squeeze() removes extra dims [1,256] → [256]
        if fusion is not None:
            if isinstance(fusion, torch.Tensor):
                m1_fusions.append(fusion.detach().float().squeeze())
            else:
                m1_fusions.append(
                    torch.tensor(fusion, dtype=torch.float32).squeeze()
                )
        else:
            m1_fusions.append(torch.zeros(256))

    m1_fp   = torch.tensor(m1_fake_probs,
                            dtype=torch.float32).to(device).unsqueeze(1)
    m1_mm   = torch.tensor(m1_mismatches,
                            dtype=torch.float32).to(device).unsqueeze(1)
    fusions = torch.stack(m1_fusions).to(device)  # [B, 256]

    # ── Module 2 ──────────────────────────────────────────────
    ev_scores = get_evidence_scores(
        texts, nli_model, nli_tokenizer, device
    )  # [B, 6]

    # ── Article node [B, 262] ─────────────────────────────────
    article_features = torch.cat([fusions, ev_scores], dim=1)

    # ── Module 3 ──────────────────────────────────────────────
    m3_fake_probs = []
    for i in range(article_features.size(0)):
        feat   = article_features[i].unsqueeze(0).cpu()
        m3_out = run_module3(article_features=feat)
        prob   = m3_out.get("fake_probability", 0.5)
        if isinstance(prob, torch.Tensor):
            prob = prob.item()
        m3_fake_probs.append(float(prob))

    m3_fp = torch.tensor(m3_fake_probs,
                          dtype=torch.float32).to(device).unsqueeze(1)

    # ── Combine → [B, 9] ──────────────────────────────────────
    combined = torch.cat([m1_fp, m1_mm, ev_scores, m3_fp], dim=1)

    # ── Classify ──────────────────────────────────────────────
    logits = classifier(combined).squeeze(1)
    loss   = loss_fn(logits, labels)
    preds  = (torch.sigmoid(logits) > 0.5).long()

    return loss, preds, labels.long()


# ================================================================
# EVALUATE
# ================================================================
def evaluate(loader, module1, nli_model, nli_tokenizer,
             classifier, loss_fn, device):
    classifier.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            loss, preds, labels = run_step(
                batch, module1, nli_model, nli_tokenizer,
                classifier, loss_fn, device
            )
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, all_preds, all_labels


# ================================================================
# MAIN
# ================================================================
def train():
    print("=" * 60)
    print("SENTINEL — Joint End-to-End Training")
    print("=" * 60)

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print("\n  Loading FakeNewsNet (PolitiFact)...")
    train_loader, val_loader, test_loader, dataset = create_data_splits(
        data_root  = str(DATA_ROOT),
        sources    = ["politifact"],
        batch_size = CONFIG["batch_size"],
        num_workers= 0
    )

    print("\n  Loading Module 1...")
    module1 = Module1()

    print("\n  Loading Module 2 NLI (fine-tuned)...")
    if not NLI_MODEL_PATH.exists():
        print("  ❌ Run training/finetune_nli.py first")
        return
    nli_tokenizer = AutoTokenizer.from_pretrained(str(NLI_MODEL_PATH))
    nli_model     = AutoModelForSequenceClassification.from_pretrained(
        str(NLI_MODEL_PATH)
    ).to(DEVICE)
    for param in nli_model.parameters():
        param.requires_grad = False
    print("  ✅ NLI loaded and frozen")

    classifier = SentinelClassifier(CONFIG["dropout"]).to(DEVICE)

    fake_count = sum(1 for s in dataset.samples if s["label"] == 1)
    real_count = sum(1 for s in dataset.samples if s["label"] == 0)
    pos_weight = torch.tensor([real_count / fake_count]).to(DEVICE)
    loss_fn    = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"\n  Fake: {fake_count} | Real: {real_count}")

    optimizer = AdamW(classifier.parameters(),
                      lr=CONFIG["learning_rate"], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    best_val_f1 = 0.0
    best_epoch  = 0
    history     = []

    print(f"\n  Training for {CONFIG['epochs']} epochs")
    print(f"  Train batches: {len(train_loader)}")
    print()

    for epoch in range(1, CONFIG["epochs"] + 1):
        classifier.train()
        train_loss = 0.0
        correct    = 0
        total      = 0

        for step, batch in enumerate(train_loader):
            loss, preds, labels = run_step(
                batch, module1, nli_model, nli_tokenizer,
                classifier, loss_fn, DEVICE
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(), 1.0
            )
            optimizer.step()

            train_loss += loss.item()
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            if (step + 1) % 10 == 0:
                print(f"  Epoch {epoch} | "
                      f"Step {step+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {correct/total*100:.1f}%")

        scheduler.step()

        val_loss, val_f1, _, _ = evaluate(
            val_loader, module1, nli_model, nli_tokenizer,
            classifier, loss_fn, DEVICE
        )

        print(f"\n  ── Epoch {epoch} Summary ──")
        print(f"  Train Loss : {train_loss/len(train_loader):.4f}")
        print(f"  Train Acc  : {correct/total*100:.1f}%")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Val F1     : {val_f1:.4f}")

        history.append({
            "epoch"     : epoch,
            "train_loss": train_loss / len(train_loader),
            "train_acc" : correct / total * 100,
            "val_loss"  : val_loss,
            "val_f1"    : val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save({
                "epoch"           : epoch,
                "classifier_state": classifier.state_dict(),
                "val_f1"          : val_f1,
                "config"          : CONFIG,
            }, SAVE_PATH / "best_model.pt")
            print(f"  ✅ Best model saved! Val F1: {val_f1:.4f}")
        print()

    # Final test
    print("=" * 60)
    print("  Final Test Evaluation")
    print("=" * 60)
    ckpt = torch.load(SAVE_PATH / "best_model.pt", map_location=DEVICE)
    classifier.load_state_dict(ckpt["classifier_state"])

    test_loss, test_f1, preds, labels_true = evaluate(
        test_loader, module1, nli_model, nli_tokenizer,
        classifier, loss_fn, DEVICE
    )

    print(f"\n  Best epoch : {best_epoch}")
    print(f"  Test Loss  : {test_loss:.4f}")
    print(f"  Test F1    : {test_f1:.4f}")
    print()
    print(classification_report(
        labels_true, preds, target_names=["Real", "Fake"]
    ))

    with open(SAVE_PATH / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ Saved to: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train()