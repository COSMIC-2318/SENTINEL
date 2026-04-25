# prompts.py

def build_pass1_prompt(article_text, module1_output, evidence_score, module3_output, has_image=True):

    # ── Image-text mismatch ───────────────────────────────────
    pm = module1_output.get('pretrained_mismatch_prob')
    if pm is not None:
        mismatch_line = (
            f"Pretrained mismatch score: {pm} (0=consistent, 1=mismatched). "
            f"{module1_output['mismatch_description']}."
        )
    else:
        mismatch_line = (
            f"Fusion mismatch score: {module1_output['attention_score']}. "
            f"{module1_output['mismatch_description']}."
        )

    # ── Claim verification ────────────────────────────────────
    claims = evidence_score.get("claims", [])
    if claims:
        claim_lines = ""
        for i, c in enumerate(claims, 1):
            claim_lines += (
                f"\n  Claim {i}: \"{c['claim']}\"\n"
                f"  Result: {c['verdict']} | Confidence: {c['confidence']}\n"
                f"  Evidence: {c['evidence'][:200]}\n"
            )
        claim_summary = (
            f"{len(claims)} claims checked — "
            f"{evidence_score['claims_supported']} supported, "
            f"{evidence_score['claims_contradicted']} contradicted, "
            f"{evidence_score['claims_neutral']} neutral.\n"
            f"{claim_lines}"
        )
    else:
        claim_summary = (
            "No verifiable factual claims could be extracted from this article. "
            "Do NOT invent, label, or reference any claims. "
            "Base the verdict on image-text mismatch and network signals only."
        )

    # ── Network / graph ───────────────────────────────────────
    ensemble_prob = module3_output.get('ensemble_prob')
    gnn_prob      = module3_output.get('fake_prob', 'N/A')

    network_lines = f"GNN fake probability score: {gnn_prob} (1.0 = likely fake, 0.0 = likely real)\n"
    if ensemble_prob is not None:
        network_lines += (
            f"Ensemble classifier score: {ensemble_prob} "
            f"(trained on FakeNewsNet+WELFake — combines all signals)\n"
        )
    network_lines += (
        "Author credibility: not available (no author metadata for this article)\n"
        "Domain trustworthiness: not available (no domain metadata for this article)"
    )

    image_section = (
        f"\nIMAGE-TEXT ANALYSIS:\n{mismatch_line}\n"
        if has_image else
        "\nIMAGE-TEXT ANALYSIS:\nNo image was provided — do not comment on image-text consistency.\n"
    )

    image_rule = (
        "2. IMAGE-TEXT mismatch score above 0.6 reinforces a Fake/Likely Fake verdict.\n"
        if has_image else ""
    )

    return f"""You are a fact-checking analyst. Analyze this article using ONLY the data provided below.
Do NOT invent claim labels, source names, or author details that are not in this data.

ARTICLE:
{article_text}
{image_section}
FACT-CHECKING RESULTS — PRIMARY SIGNAL (live Wikipedia, Tavily, Google Fact Check):
{claim_summary}

NETWORK ANALYSIS — SUPPORTING SIGNAL:
{network_lines}

VERDICT DECISION RULES (apply in order):
1. FACT-CHECKING RESULTS is the most important signal. If any claim was contradicted → lean Fake or Likely Fake. If all claims were supported → lean Real or Likely Real. If no claims were checked → rely on network signals.
{image_rule}3. Network scores (GNN, ensemble) above 0.6 add supporting evidence for Fake.
4. When signals conflict, trust FACT-CHECKING RESULTS over all others.

Write an initial verdict using the exact claim texts above — do not invent any.
Include:
1. Overall label — choose EXACTLY ONE: Fake, Likely Fake, Uncertain, Likely Real, Real
2. Which specific claims (use the exact text) were contradicted or unsupported — or state "no claims were checked"
3. What the image-text mismatch score means for this article
4. A recommended action — choose ONE: Flag, Human Review, Suppress, No Action
"""


def build_pass2_prompt(initial_verdict):
    return f"""You are a senior editorial auditor reviewing this fact-check verdict:

{initial_verdict}

Check each principle. State PASSES or FAILS and why:

PRINCIPLE 1 — Evidence Fidelity:
Does the verdict cite only evidence that was actually provided? No invented claim labels or source names?

PRINCIPLE 2 — Uncertainty Acknowledgment:
If any probability is below 0.75, does the verdict use hedged language (likely, possibly, uncertain)?

PRINCIPLE 3 — Modality Consistency:
Does the verdict mention the image-text mismatch score and what it means?

PRINCIPLE 4 — Honest Gaps:
If no claims were verified or no author data was available, does the verdict say so honestly?

PRINCIPLE 5 — Bias vs Falsehood:
Does the verdict separate factually false claims from biased-but-true framing?

For each FAIL, state exactly what to correct.
"""


def build_pass3_prompt(initial_verdict, critique):
    return f"""You wrote this initial verdict:

{initial_verdict}

You received this critique:
{critique}

Write the REVISED final verdict fixing every FAIL. Use ONLY data from the original analysis — no invented details.

Follow this EXACT format and stop after RECOMMENDED ACTION. Do not add any extra text.
Write the REASONING bullets in plain English for a general reader — no technical jargon, no signal numbers, no model names.

VERDICT: (choose ONE: Fake, Likely Fake, Uncertain, Likely Real, Real)
CONFIDENCE: (choose ONE: High, Medium, Low)

REASONING:
- (one plain-English sentence about whether the image matches the article)
- (one plain-English sentence about what the fact-checking found — name the actual claim if one was checked)
- (one plain-English sentence about the overall reliability signals)

RECOMMENDED ACTION: (choose ONE: Flag for human review, Suppress, No action, Escalate)"""
