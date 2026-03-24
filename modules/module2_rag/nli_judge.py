from transformers import pipeline

print("Loading NLI model...")
nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-roberta-base",
    device=-1
)
print("NLI model loaded.")


def judge_claim(claim, evidence_list):
    """
    Input  — a single claim + list of evidence passages
    Output — judgment for each passage + final summary
    """
    judgments = []
    for evidence in evidence_list:
        result = nli_model(f"{evidence['text']} [SEP] {claim}")
        label = result[0]['label'].upper()
        score = result[0]['score']
        judgments.append({
            "evidence":   evidence['text'],
            "label":      label,
            "confidence": round(score, 3)
        })

    labels = [j['label'] for j in judgments]

    # Overall verdict = whichever label appears most
    supported    = labels.count("ENTAILMENT")
    contradicted = labels.count("CONTRADICTION")
    neutral      = labels.count("NEUTRAL")

    if contradicted > supported:
        overall = "CONTRADICTION"
    elif supported > contradicted:
        overall = "ENTAILMENT"
    else:
        overall = "NEUTRAL"

    summary = {
        "claim":           claim,
        "supported":       supported,
        "contradicted":    contradicted,
        "neutral":         neutral,
        "overall_verdict": overall,
        "avg_confidence":  round(sum(j['confidence'] for j in judgments) / len(judgments), 3),
        "judgments":       judgments
    }
    return summary


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