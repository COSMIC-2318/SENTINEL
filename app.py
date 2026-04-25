import streamlit as st
import sys
import os
import tempfile
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import load_models, run_sentinel

st.set_page_config(page_title="SENTINEL", page_icon="🛡️", layout="centered")

@st.cache_resource
def get_models():
    return load_models()

models = get_models()


def strip_markdown(text):
    """Remove markdown formatting characters from LLM output."""
    import re
    text = re.sub(r'\*{1,3}', '', text)   # remove *, **, ***
    text = re.sub(r'#{1,6}\s*', '', text) # remove ### headers
    text = re.sub(r'`{1,3}', '', text)    # remove backticks
    return text


def parse_final_verdict(raw_text):
    """Extract structured fields from Pass 3 output."""
    text       = strip_markdown(raw_text)
    verdict    = "Unknown"
    confidence = "Unknown"
    reasoning  = []
    action     = "Unknown"

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            raw = line.split(":", 1)[1].strip()
            verdict = raw.split("/")[0].strip()
        elif line.upper().startswith("CONFIDENCE:"):
            confidence = line.split(":", 1)[1].strip().split("/")[0].strip()
        elif line.upper().startswith("RECOMMENDED ACTION:"):
            action = line.split(":", 1)[1].strip()
        elif line.startswith("- ") or line.startswith("• "):
            bullet = line.lstrip("-• ").strip()
            if ":" in bullet:
                after = bullet.split(":", 1)[1].strip()
                if after:
                    bullet = after
            reasoning.append(bullet)

    return verdict, confidence, reasoning, action


def verdict_color(verdict):
    v = verdict.upper()
    if "LIKELY FAKE" in v:  return "warning",  "⚠️"
    if "FAKE"        in v:  return "error",     "🚨"
    if "LIKELY REAL" in v:  return "info",      "✅"
    if "REAL"        in v:  return "success",   "✅"
    return "warning", "🤔"


# ── Header ────────────────────────────────────────────────────
st.title("🛡️ SENTINEL")
st.caption("Semantic Evidence Network with Temporal Intelligence for News Evaluation and Lie-detection")
st.markdown("---")

# ── Inputs ────────────────────────────────────────────────────
article_text = st.text_area(
    "📰 Paste the article text",
    height=200,
    placeholder="Paste the news article you want to analyze..."
)

uploaded_image = st.file_uploader(
    "🖼️ Article image (optional — leave blank to analyze text only)",
    type=["jpg", "jpeg", "png"]
)
if uploaded_image:
    st.image(uploaded_image, width=300)

st.markdown("---")

_, center, _ = st.columns([1, 1, 1])
with center:
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────
if analyze_clicked:

    if not article_text.strip():
        st.error("Please paste an article before analyzing.")
        st.stop()

    # Use uploaded image or fall back to blank grey image
    has_image = uploaded_image is not None
    if has_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_image.read())
            tmp_image_path = tmp.name
    else:
        blank = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            blank.save(tmp.name)
            tmp_image_path = tmp.name
    cleanup = True

    with st.spinner("Analyzing... this may take 30–60 seconds"):
        result = run_sentinel(
            article_text=article_text,
            image_path=tmp_image_path,
            has_image=has_image,
            models=models
        )

    if cleanup:
        os.unlink(tmp_image_path)

    # ── Parse Pass 3 structured output ────────────────────────
    pass3_text              = result.get("pass3", "")
    verdict, conf, reasons, action = parse_final_verdict(pass3_text)

    # Fall back to raw text if parsing failed
    if verdict == "Unknown":
        verdict = result.get("final_verdict", "Unknown")

    st.markdown("---")

    # ── Verdict ───────────────────────────────────────────────
    style, icon = verdict_color(verdict)
    getattr(st, style)(f"{icon} **{verdict}** — Confidence: {conf}")

    # ── Key Reasoning ─────────────────────────────────────────
    if reasons:
        st.markdown("**Why:**")
        for r in reasons:
            st.markdown(f"- {r}")

    # ── Recommended Action ────────────────────────────────────
    if action != "Unknown":
        st.info(f"**Recommended Action:** {action}")

    st.markdown("---")

    # ── Detailed Analysis (collapsed by default) ──────────────
    with st.expander("View Detailed Analysis"):

        # Module 1
        st.markdown("**Module 1 — Multimodal Encoder**")
        m1 = result.get("module1_output", {})
        if result.get("has_image"):
            c1, c2, c3 = st.columns(3)
            c1.metric("P(Fake)",        m1.get("fake_prob",       "N/A"))
            c2.metric("Mismatch Score", m1.get("attention_score", "N/A"))
            c3.metric("Verdict",        m1.get("verdict",         "N/A"))
            pm = m1.get("pretrained_mismatch_prob")
            if pm is not None:
                st.caption(f"Pretrained image-text mismatch: {pm} | {m1.get('mismatch_description','')}")
            else:
                st.caption(m1.get("mismatch_description", ""))
        else:
            st.caption("No image provided — image-text mismatch not evaluated. Text encoding still used for downstream modules.")

        st.markdown("---")

        # Module 2
        st.markdown("**Module 2 — Claim Verification**")
        m2 = result.get("evidence_score", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Claims",      m2.get("total_claims",       0))
        c2.metric("Supported",   m2.get("claims_supported",   0))
        c3.metric("Contradicted",m2.get("claims_contradicted",0))
        c4.metric("Neutral",     m2.get("claims_neutral",     0))
        for i, claim in enumerate(m2.get("claims", []), 1):
            v = claim.get("verdict", "NEUTRAL")
            icon = "✅" if v == "ENTAILMENT" else ("❌" if v == "CONTRADICTION" else "❓")
            st.markdown(f"{icon} **Claim {i}:** {claim.get('claim','')}")
            st.caption(f"{v} | confidence {claim.get('confidence','N/A')} | source: {claim.get('source','N/A')}")

        st.markdown("---")

        # Module 3
        st.markdown("**Module 3 — Graph Neural Network**")
        m3 = result.get("module3_output", {})
        c1, c2 = st.columns(2)
        c1.metric("GNN Fake Prob", m3.get("fake_prob", "N/A"))
        ep = m3.get("ensemble_prob")
        if ep is not None:
            c2.metric("Ensemble Score", ep)
        st.caption("Author / domain signals: not available (requires live graph data)")

        st.markdown("---")

        # Module 4 — 3 passes
        st.markdown("**Module 4 — Constitutional Adjudicator (3 passes)**")
        st.markdown("*Pass 1 — Initial verdict*")
        st.markdown(result.get("pass1", "N/A"))
        st.markdown("*Pass 2 — Self critique*")
        st.markdown(result.get("pass2_critique", "N/A"))
        st.markdown("*Pass 3 — Revised verdict*")
        st.markdown(result.get("pass3", "N/A"))

    st.caption("SENTINEL — Built by Ankit | RoBERTa · CLIP · HGT · LLaMA-3")
