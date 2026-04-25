# nli_judge.py — uses fine-tuned NLI from models/nli_finetuned/

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SENTINEL_ROOT  = Path(__file__).resolve().parent.parent.parent
NLI_MODEL_PATH = SENTINEL_ROOT / "models" / "nli_finetuned"

# Label ordering matches finetune_nli.py LIAR_TO_NLI mapping:
# true/mostly-true → 0 (ENTAILMENT)
# half-true        → 1 (NEUTRAL)
# false/pants-fire → 2 (CONTRADICTION)
ID2LABEL = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}

print("Loading NLI model...")
_tokenizer = AutoTokenizer.from_pretrained(str(NLI_MODEL_PATH))
_model = AutoModelForSequenceClassification.from_pretrained(
    str(NLI_MODEL_PATH), num_labels=3
)
_model.eval()
print("NLI model loaded.")


def judge_claim(claim, evidence_list):
    """
    Input  — a single claim + list of evidence passages
    Output — judgment for each passage + final summary
    """
    judgments = []
    for evidence in evidence_list:
        enc = _tokenizer(
            evidence['text'],
            claim,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = _model(**enc).logits
        probs   = torch.softmax(logits, dim=1)[0]
        pred_id = probs.argmax().item()
        label   = ID2LABEL[pred_id]
        score   = probs[pred_id].item()

        judgments.append({
            "evidence":   evidence['text'],
            "label":      label,
            "confidence": round(score, 3)
        })

    labels       = [j['label'] for j in judgments]
    supported    = labels.count("ENTAILMENT")
    contradicted = labels.count("CONTRADICTION")
    neutral      = labels.count("NEUTRAL")

    if contradicted > supported:
        overall = "CONTRADICTION"
    elif supported > contradicted:
        overall = "ENTAILMENT"
    else:
        overall = "NEUTRAL"

    return {
        "claim":           claim,
        "supported":       supported,
        "contradicted":    contradicted,
        "neutral":         neutral,
        "overall_verdict": overall,
        "avg_confidence":  round(
            sum(j['confidence'] for j in judgments) / len(judgments), 3
        ),
        "judgments":       judgments
    }


# --- Test ---
if __name__ == "__main__":
    claim = "NASA confirmed water on Mars"
    evidence_list = [
        {"text": "NASA scientists confirmed the detection of water molecules on the Martian surface in early 2026."},
        {"text": "Scientists confirmed that Mars has a thin atmosphere composed mostly of carbon dioxide."},
        {"text": "NASA's Perseverance rover collected rock samples from the Jezero Crater on Mars."},
    ]
    result = judge_claim(claim, evidence_list)
    print(f"Claim: {result['claim']}")
    print(f"Supported:       {result['supported']}")
    print(f"Contradicted:    {result['contradicted']}")
    print(f"Neutral:         {result['neutral']}")
    print(f"Overall verdict: {result['overall_verdict']}")
    print(f"Avg confidence:  {result['avg_confidence']}")
