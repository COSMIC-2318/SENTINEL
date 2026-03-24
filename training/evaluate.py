"""
================================================================
SENTINEL — Model Evaluation
File: training/evaluate.py

What this file does:
    Loads the best saved model from joint training and evaluates
    it on the held-out test set. Produces a full metrics report.

    This is the HONEST score — test set was never touched during
    training. This is the number you put on your resume.

Metrics produced:
    - F1 Score (macro)     → main metric for fake news detection
    - Precision & Recall   → per class breakdown
    - Accuracy             → overall correctness
    - AUC-ROC              → discrimination ability
    - Confusion Matrix     → what kinds of mistakes it makes
    - Example predictions  → qualitative inspection

Usage:
    cd ~/Ankit/SENTINEL
    conda activate sentinel_env
    python training/evaluate.py
================================================================
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)
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
MODEL_PATH     = SENTINEL_ROOT / "models" / "sentinel_joint" / "best_model.pt"
REPORT_PATH    = SENTINEL_ROOT / "models" / "sentinel_joint" / "eval_report.json"

# ── Device ────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ================================================================
# CLASSIFIER (same architecture as train.py)
# ================================================================
class SentinelClassifier(torch.nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(9, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1)
        )

    def forward(self, features):
        return self.classifier(features)


# ================================================================
# EVIDENCE SCORES (same as train.py)
# ================================================================
def get_evidence_scores(texts, nli_model, nli_tokenizer, device):
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
# PREDICT ON A SINGLE BATCH
# ================================================================
def predict_batch(batch, module1, nli_model, nli_tokenizer,
                  classifier, device):
    """
    Returns predictions, probabilities, and true labels
    for one batch — without computing loss.
    """
    texts  = batch["text"]
    images = batch["image"]
    labels = batch["label"]

    m1_fake_probs = []
    m1_mismatches = []
    m1_fusions    = []

    for i in range(len(texts)):
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
    fusions = torch.stack(m1_fusions).to(device)

    ev_scores        = get_evidence_scores(
        texts, nli_model, nli_tokenizer, device
    )
    article_features = torch.cat([fusions, ev_scores], dim=1)

    m3_fake_probs = []
    for i in range(article_features.size(0)):
        feat   = article_features[i].unsqueeze(0).cpu()
        m3_out = run_module3(article_features=feat)
        prob   = m3_out.get("fake_probability", 0.5)
        if isinstance(prob, torch.Tensor):
            prob = prob.item()
        m3_fake_probs.append(float(prob))

    m3_fp    = torch.tensor(m3_fake_probs,
                             dtype=torch.float32).to(device).unsqueeze(1)
    combined = torch.cat([m1_fp, m1_mm, ev_scores, m3_fp], dim=1)

    with torch.no_grad():
        logits = classifier(combined).squeeze(1)
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).long()

    return (
        preds.cpu().numpy(),
        probs.cpu().numpy(),
        labels.long().numpy(),
        texts,
        batch.get("title", [""] * len(texts))
    )


# ================================================================
# MAIN EVALUATION
# ================================================================
def evaluate():
    print("=" * 60)
    print("SENTINEL — Model Evaluation Report")
    print("=" * 60)

    # ── Check model exists ────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"❌ No trained model found at: {MODEL_PATH}")
        print("   Run training/train.py first.")
        return

    # ── Load dataset ──────────────────────────────────────────
    print("\n  Loading test set...")
    _, _, test_loader, dataset = create_data_splits(
        data_root  = str(DATA_ROOT),
        sources    = ["politifact"],
        batch_size = 8,
        num_workers= 0
    )
    print(f"  Test samples: {len(test_loader.dataset)}")

    # ── Load modules ──────────────────────────────────────────
    print("\n  Loading Module 1...")
    module1 = Module1()

    print("\n  Loading Module 2 NLI...")
    nli_tokenizer = AutoTokenizer.from_pretrained(str(NLI_MODEL_PATH))
    nli_model     = AutoModelForSequenceClassification.from_pretrained(
        str(NLI_MODEL_PATH)
    ).to(DEVICE)
    for param in nli_model.parameters():
        param.requires_grad = False

    # ── Load trained classifier ───────────────────────────────
    print("\n  Loading trained classifier...")
    ckpt       = torch.load(MODEL_PATH, map_location=DEVICE)
    classifier = SentinelClassifier(dropout=0.3).to(DEVICE)
    classifier.load_state_dict(ckpt["classifier_state"])
    classifier.eval()
    print(f"  ✅ Loaded from epoch {ckpt['epoch']} "
          f"(val F1: {ckpt['val_f1']:.4f})")

    # ── Run evaluation ────────────────────────────────────────
    print("\n  Running evaluation on test set...")
    all_preds  = []
    all_probs  = []
    all_labels = []
    all_texts  = []
    all_titles = []

    for batch in test_loader:
        preds, probs, labels, texts, titles = predict_batch(
            batch, module1, nli_model, nli_tokenizer,
            classifier, DEVICE
        )
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels)
        all_texts.extend(texts)
        all_titles.extend(titles)

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ================================================================
    # METRICS
    # ================================================================
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    # Core metrics
    accuracy  = accuracy_score(all_labels, all_preds)
    f1_macro  = f1_score(all_labels, all_preds, average="macro")
    f1_fake   = f1_score(all_labels, all_preds, pos_label=1,
                          average="binary")
    f1_real   = f1_score(all_labels, all_preds, pos_label=0,
                          average="binary")
    precision = precision_score(all_labels, all_preds, average="macro")
    recall    = recall_score(all_labels, all_preds, average="macro")

    # AUC-ROC
    # Measures ability to distinguish fake from real across ALL
    # thresholds — not just 0.5. Perfect = 1.0, random = 0.5
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc_roc = 0.0

    # Confusion matrix
    # [[TN, FP],   TN = correctly predicted real
    #  [FN, TP]]   TP = correctly predicted fake
    #              FP = real article predicted as fake
    #              FN = fake article missed (most dangerous)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Accuracy       : {accuracy:.4f}        │")
    print(f"  │  F1 Macro       : {f1_macro:.4f}        │")
    print(f"  │  F1 Fake        : {f1_fake:.4f}        │")
    print(f"  │  F1 Real        : {f1_real:.4f}        │")
    print(f"  │  Precision Macro: {precision:.4f}        │")
    print(f"  │  Recall Macro   : {recall:.4f}        │")
    print(f"  │  AUC-ROC        : {auc_roc:.4f}        │")
    print(f"  └─────────────────────────────────┘")

    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Real   Fake")
    print(f"  Actual Real  [{cm[0][0]:4d}]  [{cm[0][1]:4d}]")
    print(f"  Actual Fake  [{cm[1][0]:4d}]  [{cm[1][1]:4d}]")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  TN (real→real)  : {tn}  ✅")
    print(f"  TP (fake→fake)  : {tp}  ✅")
    print(f"  FP (real→fake)  : {fp}  ⚠️  False alarm")
    print(f"  FN (fake→real)  : {fn}  ❌ Missed fake news")

    print(f"\n  Full Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Real", "Fake"]
    ))

    # ================================================================
    # EXAMPLE PREDICTIONS
    # ================================================================
    print("=" * 60)
    print("  EXAMPLE PREDICTIONS")
    print("=" * 60)

    # Show 3 correct and 3 incorrect predictions
    correct_idx   = np.where(all_preds == all_labels)[0]
    incorrect_idx = np.where(all_preds != all_labels)[0]

    print("\n  ✅ Correctly Classified Examples:")
    for idx in correct_idx[:3]:
        label_name = "FAKE" if all_labels[idx] == 1 else "REAL"
        pred_name  = "FAKE" if all_preds[idx]  == 1 else "REAL"
        title      = all_titles[idx] if all_titles[idx] else "No title"
        print(f"\n  [{label_name}→{pred_name}] conf:{all_probs[idx]:.2f}")
        print(f"  {title[:80]}...")

    print("\n  ❌ Incorrectly Classified Examples:")
    for idx in incorrect_idx[:3]:
        label_name = "FAKE" if all_labels[idx] == 1 else "REAL"
        pred_name  = "FAKE" if all_preds[idx]  == 1 else "REAL"
        title      = all_titles[idx] if all_titles[idx] else "No title"
        print(f"\n  [{label_name}→{pred_name}] conf:{all_probs[idx]:.2f}")
        print(f"  {title[:80]}...")

    # ================================================================
    # SAVE REPORT
    # ================================================================
    report = {
        "accuracy"        : float(accuracy),
        "f1_macro"        : float(f1_macro),
        "f1_fake"         : float(f1_fake),
        "f1_real"         : float(f1_real),
        "precision_macro" : float(precision),
        "recall_macro"    : float(recall),
        "auc_roc"         : float(auc_roc),
        "confusion_matrix": cm.tolist(),
        "true_negatives"  : int(tn),
        "false_positives" : int(fp),
        "false_negatives" : int(fn),
        "true_positives"  : int(tp),
        "test_size"       : len(all_labels),
        "model_epoch"     : int(ckpt["epoch"]),
        "model_val_f1"    : float(ckpt["val_f1"]),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  ✅ Report saved to: {REPORT_PATH}")
    print(f"\n  KEY NUMBER FOR RESUME:")
    print(f"  Test F1 (macro): {f1_macro:.4f}")
    print(f"  AUC-ROC        : {auc_roc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()