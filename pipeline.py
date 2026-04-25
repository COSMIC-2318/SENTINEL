# pipeline.py — SENTINEL End-to-End Pipeline
# Wires Module 1 → 2 → 3 → 4 into a single function call

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

SENTINEL_ROOT     = Path(__file__).resolve().parent
CLASSIFIER_PATH   = SENTINEL_ROOT / "models" / "sentinel_joint_1" / "best_model.pt"


class _SentinelClassifier(nn.Module):
    """Matches training/train.py SentinelClassifier (9→32→1)."""
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.classifier(x)


def _load_ensemble_classifier():
    if not CLASSIFIER_PATH.exists():
        print(f"  [Pipeline] Ensemble classifier not found: {CLASSIFIER_PATH}")
        return None
    clf = _SentinelClassifier()
    ckpt = torch.load(str(CLASSIFIER_PATH), map_location="cpu")
    clf.load_state_dict(ckpt["classifier_state"])
    clf.eval()
    print(f"  [Pipeline] Ensemble classifier loaded (Val F1={ckpt.get('val_f1', '?'):.3f})")
    return clf

# ─────────────────────────────────────────
# Tell Python where each module lives
# ─────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules/module1_multimodal'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules/module2_rag'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules/module3_gnn'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules/module4_adjudicator'))

# ─────────────────────────────────────────
# Import each module's main function
# ─────────────────────────────────────────
from module1 import Module1
from module2 import run_module2
from module3 import run_module3
from adjudicator import constitutional_adjudicate


def load_models():
    """
    Load all heavy models once.
    Called by Streamlit's @st.cache_resource decorator.
    Returns a dict so run_sentinel() can access them.
    """
    print("Initializing SENTINEL pipeline...")
    module1    = Module1()
    ensemble   = _load_ensemble_classifier()
    print("Pipeline ready.")
    return {"module1": module1, "ensemble": ensemble}


def run_sentinel(article_text: str, image_path: str, models: dict, has_image: bool = True) -> dict:
    """
    Main SENTINEL pipeline function.

    Input:
        article_text — full article text as string
        image_path   — path to article image file
        models       — dict returned by load_models()

    Output:
        structured dict with all 4 module outputs
    """

    module1  = models["module1"]
    ensemble = models.get("ensemble")

    print("=" * 60)
    print("SENTINEL — Starting Analysis")
    print("=" * 60)

    # ─────────────────────────────────────────
    # MODULE 1 — Multimodal Encoder
    # ─────────────────────────────────────────
    print("\n[MODULE 1] Multimodal Encoding...")
    image = Image.open(image_path).convert("RGB")
    m1_output = module1.predict(article_text, image)

    print(f"  Verdict:          {m1_output['verdict']}")
    print(f"  P(Fake):          {m1_output['p_fake']}")
    print(f"  Attention Score:  {m1_output['attention_score']}")
    print(f"  Mismatch:         {m1_output['mismatch_description']}")

    # ─────────────────────────────────────────
    # MODULE 2 — RAG Claim Verifier
    # ─────────────────────────────────────────
    print("\n[MODULE 2] Claim Verification...")
    m2_output = run_module2(article_text)

    if m2_output is None:
        m2_output = {
            "total_claims": 0,
            "claims_supported": 0,
            "claims_contradicted": 0,
            "claims_neutral": 0,
            "claims_filtered": 0,
            "avg_confidence": 0.0,
            "claims": []
        }

    print(f"  Total Claims:     {m2_output['total_claims']}")
    print(f"  Supported:        {m2_output['claims_supported']}")
    print(f"  Contradicted:     {m2_output['claims_contradicted']}")
    print(f"  Avg Confidence:   {m2_output['avg_confidence']}")

    # ─────────────────────────────────────────
    # BUILD 262-DIM ARTICLE FEATURE TENSOR
    # ─────────────────────────────────────────
    evidence_scores = torch.tensor([[
        float(m2_output['claims_supported']),
        float(m2_output['claims_contradicted']),
        float(m2_output['claims_neutral']),
        float(m2_output['claims_filtered']),
        float(m2_output['avg_confidence']),
        float(m2_output['total_claims'])
    ]])  # shape [1, 6]

    article_features = torch.cat(
        [m1_output['fusion_vector'], evidence_scores], dim=1
    )  # shape [1, 262]

    print(f"\n[PIPELINE] Article feature tensor built: {article_features.shape}")

    # ─────────────────────────────────────────
    # MODULE 3 — Heterogeneous GNN
    # ─────────────────────────────────────────
    print("\n[MODULE 3] Graph Neural Network...")
    m3_output = run_module3(article_features=article_features)

    print(f"  Verdict:          {m3_output['verdict']}")
    print(f"  Fake Probability: {m3_output['fake_prob']}")

    # ─────────────────────────────────────────
    # ENSEMBLE CLASSIFIER — aggregated signal
    # ─────────────────────────────────────────
    ensemble_prob = None
    if ensemble is not None:
        features = torch.tensor([[
            float(m1_output['p_fake']),
            float(m1_output['attention_score']),
            float(m2_output['claims_supported']),
            float(m2_output['claims_contradicted']),
            float(m2_output['claims_neutral']),
            float(m2_output['claims_filtered']),
            float(m2_output['avg_confidence']),
            float(m2_output['total_claims']),
            float(m3_output['fake_prob']),
        ]])
        with torch.no_grad():
            ensemble_prob = round(
                torch.sigmoid(ensemble(features)).item(), 4
            )
        print(f"\n[ENSEMBLE] Aggregated fake probability: {ensemble_prob}")

    # ─────────────────────────────────────────
    # MODULE 4 — Constitutional Adjudicator
    # ─────────────────────────────────────────
    print("\n[MODULE 4] Constitutional Adjudication...")

    module1_for_m4 = {
        "fake_prob":                m1_output['p_fake'],
        "attention_score":          m1_output['attention_score'] if has_image else None,
        "mismatch_description":     m1_output['mismatch_description'] if has_image else None,
        "pretrained_mismatch_prob": m1_output.get('pretrained_mismatch_prob') if has_image else None,
    }

    module3_for_m4 = {
        "fake_prob":     m3_output['fake_prob'],
        "author_flag":   m3_output['author_flag'],
        "domain_flag":   m3_output['domain_flag'],
        "claim_overlap": m3_output['claim_overlap'],
        "ensemble_prob": ensemble_prob,
    }

    final_result = constitutional_adjudicate(
        article_text=article_text,
        module1_output=module1_for_m4,
        evidence_score=m2_output,
        module3_output=module3_for_m4,
        has_image=has_image
    )

    print("\n" + "=" * 60)
    print("SENTINEL — FINAL VERDICT")
    print("=" * 60)
    print(final_result['final_verdict'])

    # ─────────────────────────────────────────
    # FIXED RETURN — pack ALL module outputs
    # so app.py can display each module's results
    # ─────────────────────────────────────────
    return {
        # Top-level verdict (used for the big verdict box)
        "final_verdict":  final_result["final_verdict"],
        "has_image":      has_image,

        # Module 4 — the 3 passes
        "pass1":          final_result["initial_verdict"],
        "pass2_critique": final_result["critique"],
        "pass3":          final_result["final_verdict"],

        # Module 1 — multimodal signals
        "module1_output": {
            "fake_prob":            m1_output["p_fake"],
            "attention_score":      m1_output["attention_score"],
            "verdict":              m1_output["verdict"],
            "mismatch_description": m1_output["mismatch_description"]
        },

        # Module 2 — claim verification
        "evidence_score": m2_output,

        # Module 3 — graph signals
        "module3_output": {
            "fake_prob":     m3_output["fake_prob"],
            "author_flag":   m3_output["author_flag"],
            "domain_flag":   m3_output["domain_flag"],
            "claim_overlap": m3_output["claim_overlap"]
        }
    }


# ─────────────────────────────────────────
# Test — only runs when called directly
# python pipeline.py
# NOT when imported by Streamlit
# ─────────────────────────────────────────
if __name__ == "__main__":

    test_article = """
    NASA has confirmed the discovery of liquid water on the surface 
    of Mars, according to a press release issued today. Scientists 
    say this could support microbial life. The agency plans to send 
    a crewed mission by 2027.
    """

    test_image = "notebooks/test_image.jpg"

    models = load_models()

    result = run_sentinel(
        article_text=test_article,
        image_path=test_image,
        models=models
    )

    # Verify all keys are present
    print("\nReturn keys:", list(result.keys()))