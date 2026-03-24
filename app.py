# app.py — SENTINEL Streamlit Demo UI
# Run with: streamlit run app.py

import streamlit as st
import sys
import os
import tempfile
from PIL import Image

# ─────────────────────────────────────────
# Tell Python where pipeline.py lives
# ─────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import load_models, run_sentinel

# ─────────────────────────────────────────
# PAGE CONFIG — must be the very first
# Streamlit call in the script
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODELS ONCE
# @st.cache_resource means:
#   - First visit → calls load_models()
#   - Every click after → uses cached version
#   - RoBERTa + CLIP never reload
# ─────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()

models = get_models()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🛡️ SENTINEL")
st.markdown("**Semantic Evidence Network with Temporal Intelligence for News Evaluation and Lie-detection**")
st.markdown("*A 4-module multimodal fake news detection system*")
st.markdown("---")

# ─────────────────────────────────────────
# INPUT SECTION
# Two columns side by side:
#   Left  → article text box
#   Right → image uploader
# ─────────────────────────────────────────
col1, col2 = st.columns([2, 1])  # left column is 2x wider

with col1:
    st.subheader("📰 Article Text")
    article_text = st.text_area(
        label="Paste the full article here",
        height=300,
        placeholder="Paste the news article you want to analyze..."
    )

with col2:
    st.subheader("🖼️ Article Image")
    uploaded_image = st.file_uploader(
        label="Upload the article's image",
        type=["jpg", "jpeg", "png"]
    )
    # Show a preview if image is uploaded
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

st.markdown("---")

# ─────────────────────────────────────────
# ANALYZE BUTTON
# Centered using columns trick
# ─────────────────────────────────────────
_, center, _ = st.columns([1, 1, 1])
with center:
    analyze_clicked = st.button("🔍 Analyze Article", use_container_width=True)

# ─────────────────────────────────────────
# ANALYSIS — only runs when button clicked
# ─────────────────────────────────────────
if analyze_clicked:

    # ── Input validation ──
    if not article_text.strip():
        st.error("⚠️ Please paste an article before analyzing.")
        st.stop()

    if uploaded_image is None:
        st.error("⚠️ Please upload an image before analyzing.")
        st.stop()

    # ── Save uploaded image to a temp file ──
    # run_sentinel() needs a file PATH, not a file object
    # tempfile creates a real file on disk temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_image.read())
        tmp_image_path = tmp.name

    # ── Run the full pipeline with a spinner ──
    with st.spinner("🛡️ SENTINEL is analyzing... this may take 30-60 seconds"):
        result = run_sentinel(
            article_text=article_text,
            image_path=tmp_image_path,
            models=models
        )

    # ── Clean up temp file ──
    os.unlink(tmp_image_path)

    st.markdown("---")
    st.header("📊 Analysis Results")

    # ─────────────────────────────────────────
    # FINAL VERDICT — big and prominent
    # ─────────────────────────────────────────
    verdict = result.get("final_verdict", "Unknown")

    # Color the verdict box based on result
    if "FAKE" in verdict.upper():
        st.error(f"🚨 Final Verdict: **{verdict}**")
    elif "LIKELY FAKE" in verdict.upper():
        st.warning(f"⚠️ Final Verdict: **{verdict}**")
    elif "UNCERTAIN" in verdict.upper():
        st.warning(f"🤔 Final Verdict: **{verdict}**")
    elif "LIKELY REAL" in verdict.upper():
        st.info(f"✅ Final Verdict: **{verdict}**")
    else:
        st.success(f"✅ Final Verdict: **{verdict}**")

    st.markdown("---")

    # ─────────────────────────────────────────
    # MODULE RESULTS — 4 expandable sections
    # Each module gets its own collapsible box
    # ─────────────────────────────────────────

    # ── Module 1 ──
    with st.expander("🖼️ Module 1 — Multimodal Encoder", expanded=True):
        m1 = result.get("module1_output", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("P(Fake)", f"{m1.get('fake_prob', 'N/A')}")
        c2.metric("Attention Score", f"{m1.get('attention_score', 'N/A')}")
        c3.metric("Verdict", m1.get("verdict", "N/A"))
        st.caption(f"Mismatch Description: {m1.get('mismatch_description', 'N/A')}")

    # ── Module 2 ──
    with st.expander("🔍 Module 2 — RAG Claim Verifier", expanded=True):
        m2 = result.get("evidence_score", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Claims", m2.get("total_claims", 0))
        c2.metric("✅ Supported", m2.get("claims_supported", 0))
        c3.metric("❌ Contradicted", m2.get("claims_contradicted", 0))
        c4.metric("❓ Neutral", m2.get("claims_neutral", 0))

        # Show individual claim verdicts if available
        claims = m2.get("claims", [])
        if claims:
            st.markdown("**Claim-by-Claim Breakdown:**")
            for i, claim in enumerate(claims):
                verdict_label = claim.get("verdict", "NEUTRAL")
                icon = "✅" if verdict_label == "ENTAILMENT" else ("❌" if verdict_label == "CONTRADICTION" else "❓")
                st.markdown(f"{icon} **Claim {i+1}:** {claim.get('claim_text', '')}")
                st.caption(f"Verdict: {verdict_label} | Confidence: {claim.get('confidence', 'N/A')}")

    # ── Module 3 ──
    with st.expander("🕸️ Module 3 — Graph Neural Network", expanded=True):
        m3 = result.get("module3_output", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GNN Fake Prob", f"{m3.get('fake_prob', 'N/A')}")
        c2.metric("Author Flag", "🚩 Yes" if m3.get("author_flag") else "✅ No")
        c3.metric("Domain Flag", "🚩 Yes" if m3.get("domain_flag") else "✅ No")
        c4.metric("Claim Overlap", f"{m3.get('claim_overlap', 'N/A')}")

    # ── Module 4 ──
    with st.expander("⚖️ Module 4 — Constitutional Adjudicator", expanded=True):
        st.markdown("**Pass 1 — Initial Verdict:**")
        st.info(result.get("pass1", "Not available"))

        st.markdown("**Pass 2 — Self Critique:**")
        st.warning(result.get("pass2_critique", "Not available"))

        st.markdown("**Pass 3 — Revised Final Verdict:**")
        st.success(result.get("pass3", "Not available"))

    st.markdown("---")
    st.caption("SENTINEL — Built by Ankit | Powered by RoBERTa + CLIP + HGT + LLaMA-3")