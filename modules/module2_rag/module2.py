# module2.py

from claim_extractor import extract_claims
from retriever import retrieve
from nli_judge import judge_claim


def run_module2(article_text):
    """
    Input  — full article text
    Output — evidence score vector for the whole article
    """

    # Step 1 — Extract claims
    print("\n--- Step 1: Extracting Claims ---")
    claims = extract_claims(article_text)
    print(f"Found {len(claims)} claims:")
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c}")

    if len(claims) == 0:
        print("No claims found — article may be opinion only.")
        return None

    # Step 3 — For each claim, retrieve evidence then judge it
    print("\n--- Step 2: Retrieving Evidence + NLI Judgment ---")

    total_supported    = 0
    total_contradicted = 0
    total_neutral      = 0
    total_filtered     = 0
    all_confidences    = []
    claims_detail      = []

    for claim in claims:
        print(f"\nProcessing: {claim}")

        # Retrieve top 5 relevant passages from 3 live sources
        ranked_passages = retrieve(claim)

        # No relevant evidence found
        if len(ranked_passages) == 0:
            print("  No relevant evidence found — claim unverifiable.")
            total_neutral += 1
            claims_detail.append({
                "claim":      claim,
                "verdict":    "NEUTRAL",
                "evidence":   "No relevant evidence found from any source",
                "confidence": 0.0
            })
            continue

        # Convert to format nli_judge expects
        relevant_evidence = [
            {"text": p["text"], "similarity": p["relevance_score"], "filtered": False}
            for p in ranked_passages
        ]

        # Run NLI on relevant evidence
        result = judge_claim(claim, relevant_evidence)

        total_supported    += result['supported']
        total_contradicted += result['contradicted']
        total_neutral      += result['neutral']
        all_confidences.append(result['avg_confidence'])

        claims_detail.append({
            "claim":      claim,
            "verdict":    result['overall_verdict'],
            "evidence":   ranked_passages[0]['text'],
            "source":     ranked_passages[0]['source'],
            "confidence": result['avg_confidence']
        })

    # Build final evidence score vector
    avg_confidence = round(
        sum(all_confidences) / len(all_confidences), 3
    ) if all_confidences else 0.0

    evidence_vector = {
        "total_claims":        len(claims),
        "claims_supported":    total_supported,
        "claims_contradicted": total_contradicted,
        "claims_neutral":      total_neutral,
        "claims_filtered":     total_filtered,
        "avg_confidence":      avg_confidence,
        "claims":              claims_detail
    }

    print("\n--- Final Evidence Score Vector ---")
    for key, value in evidence_vector.items():
        if key != "claims":
            print(f"  {key}: {value}")

    print("\n--- Per-Claim Breakdown ---")
    for i, c in enumerate(claims_detail, 1):
        print(f"  Claim {i}: {c['claim']}")
        print(f"    Verdict:    {c['verdict']}")
        print(f"    Source:     {c.get('source', 'N/A')}")
        print(f"    Evidence:   {c['evidence'][:80]}...")
        print(f"    Confidence: {c['confidence']}")

    return evidence_vector


# --- Test ---
if __name__ == "__main__":
    test_article = """
    The situation is extremely worrying and people are scared.
    NASA confirmed the discovery of water on Mars on March 12, 2026.
    This is a very shocking development for everyone.
    Prime Minister Modi announced a new economic policy in New Delhi yesterday.
    People are feeling uncertain about what comes next.
    """

    result = run_module2(test_article)