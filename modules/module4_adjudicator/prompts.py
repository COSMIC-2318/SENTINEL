# prompts.py

def build_pass1_prompt(article_text, module1_output, evidence_score, module3_output):

    # Format each claim
    claims_breakdown = ""
    for i, c in enumerate(evidence_score["claims"], 1):
        claims_breakdown += f"""
  Claim {i}: "{c['claim']}"
  Verdict:    {c['verdict']}
  Evidence:   {c['evidence']}
  Confidence: {c['confidence']}
"""

    return f"""
You are a fact-checking analyst. You have been given the following signals about a news article:

ARTICLE TEXT:
{article_text}

SIGNAL 1 — Multimodal Analysis:
Fake Probability (text + image): {module1_output['fake_prob']}
Image-Text Mismatch Score: {module1_output['attention_score']}
Mismatch Assessment: {module1_output['mismatch_description']}

SIGNAL 2 — Claim Verification (RAG):
Total claims extracted: {len(evidence_score['claims'])}
Average NLI confidence: {evidence_score['avg_confidence']}

Per-claim breakdown:
{claims_breakdown}

SIGNAL 3 — Graph Analysis:
Fake Probability (graph): {module3_output['fake_prob']}
Author History: {module3_output['author_flag']}
Domain Trustworthiness: {module3_output['domain_flag']}
Claim Network: {module3_output['claim_overlap']}

Based on all three signals, write an initial verdict. Include:
1. Overall assessment (Fake / Likely Fake / Uncertain / Likely Real / Real)
2. Name the SPECIFIC claims that were contradicted or unsupported
3. Reference the image-text mismatch finding explicitly
4. Reference the author and domain red flags explicitly
5. A recommended action (Flag / Human Review / Suppress / No Action)
"""


def build_pass2_prompt(initial_verdict):
    return f"""
You are a senior editorial auditor reviewing the following fact-check verdict:

{initial_verdict}

Critique this verdict against these 5 constitutional principles.
For each principle, state PASSES or FAILS and why:

PRINCIPLE 1 — Evidence Fidelity:
Does the verdict name the specific claims that were contradicted or unsupported?

PRINCIPLE 2 — Uncertainty Acknowledgment:
If any fake probability was below 0.75, does the verdict acknowledge uncertainty
instead of stating a definitive conclusion?

PRINCIPLE 3 — Modality Consistency:
Does the verdict explicitly mention the image-text mismatch finding and its score?

PRINCIPLE 4 — Graph Coherence:
Does the verdict mention the author history, domain age, and claim network overlap?

PRINCIPLE 5 — Bias vs Falsehood:
Does the verdict distinguish between factually false claims and factually accurate
claims presented with bias?

For each FAIL, state exactly what must be corrected in the revised verdict.
"""


def build_pass3_prompt(initial_verdict, critique):
    return f"""
You are a fact-checking analyst. You wrote the following initial verdict:

{initial_verdict}

You then received this critique:
{critique}

Now write a REVISED final verdict that addresses every FAIL from the critique.

Your output must follow this exact structure:

VERDICT: [Fake / Likely Fake / Uncertain / Likely Real / Real]
CONFIDENCE: [High / Medium / Low]

REASONING:
- Multimodal signal: [specific mismatch finding with score]
- Claim verification: [name each contradicted/unsupported claim with its evidence]
- Graph signal: [author history + domain age + claim overlap]

RECOMMENDED ACTION: [Flag for human review / Suppress / No action / Escalate]
"""