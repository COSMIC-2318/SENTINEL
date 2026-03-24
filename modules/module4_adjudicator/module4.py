# module4.py

from adjudicator import constitutional_adjudicate

article_text = """
NASA has confirmed the discovery of liquid water on the surface of Mars,
according to a press release issued today. Scientists say this could support
microbial life. The agency plans to send a crewed mission by 2027.
"""

# --- Module 1 output (enriched) ---
module1_output = {
    "fake_prob": 0.71,
    "attention_score": 0.73,
    "mismatch_description": "High — image shows dry barren Mars surface, inconsistent with article claim of liquid water presence"
}

# --- Module 2 output (already enriched last session) ---
evidence_score = {
    "total_claims": 3,
    "avg_confidence": 0.68,
    "claims": [
        {
            "claim": "NASA confirmed the discovery of liquid water on the surface of Mars",
            "verdict": "CONTRADICTION",
            "evidence": "Wikipedia states NASA found evidence of ancient water, not current liquid surface water",
            "confidence": 0.81
        },
        {
            "claim": "The agency plans to send a crewed mission by 2027",
            "verdict": "NEUTRAL",
            "evidence": "No Wikipedia evidence found for a 2027 crewed Mars mission",
            "confidence": 0.61
        },
        {
            "claim": "Scientists say this could support microbial life",
            "verdict": "ENTAILMENT",
            "evidence": "Wikipedia confirms scientific consensus that liquid water could support microbial life",
            "confidence": 0.74
        }
    ]
}

# --- Module 3 output (enriched) ---
module3_output = {
    "fake_prob": 0.65,
    "author_flag": "Author has 3 previously flagged articles in training data",
    "domain_flag": "Publishing domain registered 18 days ago — below credibility threshold",
    "claim_overlap": "Core claim shared with 2 other articles independently flagged as fake"
}

# --- Run the 3-pass Constitutional Adjudicator ---
result = constitutional_adjudicate(
    article_text=article_text,
    module1_output=module1_output,
    evidence_score=evidence_score,
    module3_output=module3_output
)

print("\n========== FINAL VERDICT ==========")
print(result["final_verdict"])