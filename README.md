# 🛡️ S.E.N.T.I.N.E.L.
### Semantic Evidence Network with Temporal Intelligence for News Evaluation and Lie-detection

> *A multi-signal fake news detection system that simultaneously reasons about what an article says, what it shows, what it contradicts in external knowledge, and how it relates to other articles and authors — producing a calibrated, human-readable verdict with traceable evidence.*

---

![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧭 The Problem

Most fake news detection systems read only the text of an article. But fake news doesn't always lie in the words.

A manipulated article about a protest might use a photograph from a completely different event five years ago. The text is internally consistent and well-written. A text-only classifier misses this entirely — **the lie is in the mismatch between what is claimed and what is shown.**

Studies on the NewsCLIPpings dataset show that image-text inconsistency is present in over **40% of fake news articles** that otherwise pass text-only classifiers.

Beyond that, a single article doesn't exist in isolation. It was written by someone with a credibility history, published on a domain with a trust score, and makes claims that overlap with or contradict other articles. All of this relational context is invisible to systems that analyze one article at a time.

**SENTINEL is built to address all of this simultaneously.**

---

## 🏗️ Architecture — 4 Modules

```
Article Text + Image
        │
        ▼
┌───────────────────┐
│    MODULE 1       │  ← RoBERTa (text) + CLIP (image)
│ Multimodal        │    Cross-Modal Attention Fusion
│ Evidence Encoder  │    Output: 256-dim fused vector + fake probability
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    MODULE 2       │  ← SpaCy NER (claim extraction)
│ RAG Claim         │    DPR + FAISS (semantic retrieval)
│ Verifier          │    NLI — RoBERTa-large (ENTAILMENT / CONTRADICTION / NEUTRAL)
└────────┬──────────┘    Live Sources: Wikipedia API + Tavily + Google Fact Check
         │
         ▼
┌───────────────────┐
│    MODULE 3       │  ← Heterogeneous Graph (5 node types, 8 edge types)
│ Graph Neural      │    Heterogeneous Graph Transformer (HGT)
│ Network           │    Detects coordinated inauthentic behavior, suspicious authors
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    MODULE 4       │  ← LLaMA-3 8B (via Ollama, 4-bit quantized)
│ Constitutional    │    3-pass self-critique (Constitutional AI — Anthropic, 2022)
│ Adjudicator       │    Output: Human-readable verdict with full reasoning
└───────────────────┘
         │
         ▼
  Final Verdict:
  Fake / Likely Fake / Uncertain / Likely Real / Real
  + Evidence breakdown + Recommended action
```

---

## 🔬 Module Details

### Module 1 — Multimodal Evidence Encoder

| Component | Choice | Why |
|---|---|---|
| Text Encoder | RoBERTa-base | Outperforms BERT by 4–7 GLUE points. Better at nuanced language — sarcasm, hedging, emotional amplification |
| Image Encoder | CLIP ViT-B/32 | Trained on 400M image-text pairs via contrastive learning. Image embeddings are semantically aligned with language |
| Fusion | Cross-Modal Attention | Query from text branch, Key/Value from image branch. Text tokens "look at" image patches and update themselves based on what they find — not naive concatenation |

**What it detects:** recycled images from past events, images from wrong locations, emotionally manipulative visuals paired with neutral text.

---

### Module 2 — RAG Knowledge-Grounded Claim Verifier

**Pipeline:**
1. **Claim Extraction** — SpaCy NER identifies sentences containing named entities + assertive verbs (confirmed, announced, killed, signed)
2. **Dense Passage Retrieval** — DPR bi-encoder embeds claims and evidence into the same vector space. Relevance by dot product — semantic, not keyword-based.
3. **NLI Scoring** — RoBERTa-large (MultiNLI + FEVER) classifies each claim-evidence pair as ENTAILMENT / CONTRADICTION / NEUTRAL

**Live Evidence Sources:**
- Wikipedia Live API — established background knowledge
- Tavily Search API — live web consensus, breaking news
- Google Fact Check API — pre-verified claims from PolitiFact, Snopes, AFP

**Key insight:** "No evidence found" is not a neutral result. Fake news often makes deliberately unverifiable claims. The absence of retrievable evidence is itself a red flag signal.

**Output:** Structured evidence vector — claims supported / contradicted / unverifiable / average NLI confidence

---

### Module 3 — Heterogeneous Graph Neural Network

**Node Types:**
- `Article` — multimodal embedding + evidence scores + publication timestamp
- `Author` — historical fake/real ratio, account age, emotional language score
- `Domain` — domain age, traffic rank, historical credibility, misinformation list membership
- `Claim` — DPR embedding, NLI verdict distribution, cross-article frequency
- `Entity` — named entities with learned suspicion weights from training data

**Edge Types:** `published_by`, `authored_by`, `makes_claim`, `shares_claim_with`, `contradicts`, `mentions_entity`, `co_mentions` + all reverse edges

**Model:** Heterogeneous Graph Transformer (HGT) — learns separate attention parameters per edge type. `authored_by` and `contradicts` edges carry fundamentally different information and are treated as such.

**What it detects:** coordinated inauthentic behavior (multiple fake articles making the same false claim), suspicious author patterns, low-credibility publishing domains.

---

### Module 4 — Constitutional Adjudicator

**Architecture:** LLaMA-3 8B running locally via Ollama (4-bit quantized), applying Anthropic's Constitutional AI methodology to fake news adjudication.

**3-Pass Self-Critique:**
- **Pass 1** — Initial verdict with step-by-step reasoning
- **Pass 2** — Self-critique against 5 constitutional principles:
  - Evidence Fidelity — verdict must not contradict NLI-verified claims
  - Uncertainty Acknowledgment — hedged probability → hedged language
  - Modality Consistency — high image-text mismatch must be mentioned
  - Graph Coherence — suspicious author history must be flagged
  - Bias vs. Falsehood — factually false ≠ factually accurate but biased
- **Pass 3** — Revised verdict incorporating critique

**Output:** `Fake / Likely Fake / Uncertain / Likely Real / Real` + verified claims list + visual evidence assessment + author credibility signal + recommended action (flag / human review / suppress / no action)

---

## ⚙️ Pipeline Integration

```python
# End-to-end usage
from pipeline import run_sentinel

result = run_sentinel(
    article_text="Your article text here...",
    image_path="path/to/article/image.jpg"
)

print(result["final_verdict"])
print(result["reasoning"])
```

**The 262-dim article node** assembled in pipeline.py:
```
torch.cat([fusion_vector (256-dim), evidence_scores (6-dim)])
```

Module 3 receives real article features from Modules 1 and 2 — not hardcoded random values. This is the core integration responsibility of `pipeline.py`.

---

## 🚀 Current Status

| Component | Status | Notes |
|---|---|---|
| Module 1 — Multimodal Encoder | ✅ Complete | RoBERTa + CLIP + Cross-Modal Attention |
| Module 2 — RAG Claim Verifier | ✅ Complete | Upgraded to live 3-source retrieval |
| Module 3 — Heterogeneous GNN | ✅ Complete | HGT with 5 node types, 8 edge types |
| Module 4 — Constitutional Adjudicator | ✅ Complete | LLaMA-3 8B, 3-pass self-critique |
| Pipeline — End-to-End Integration | ✅ Complete | All 4 modules connected and working |
| Streamlit Demo UI | ✅ Complete | Interactive web interface |
| Training on FakeNewsNet | 🔄 Next Phase | Currently running on pretrained weights |
| Ablation Studies | 🔄 Planned | Each module's contribution to be measured |

> **Honest note:** The architecture, pipeline, and Streamlit demo are fully working. The system currently runs on pretrained weights (RoBERTa, CLIP, LLaMA-3) without fine-tuning on FakeNewsNet. End-to-end training on FakeNewsNet is the immediate next phase of development.

---

## 📦 Setup

### Prerequisites
- Python 3.10
- Conda
- Ollama (for LLaMA-3 locally)
- Mac M-series or CUDA GPU recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SENTINEL.git
cd SENTINEL

# Create and activate environment
conda create -n sentinel_env python=3.10
conda activate sentinel_env

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Pull LLaMA-3 via Ollama
ollama pull llama3
```

### API Keys

Create a `.env` file in the project root:

```
TAVILY_API_KEY=your_tavily_key_here
GOOGLE_FACT_CHECK_KEY=your_google_key_here
```

> **Never commit your `.env` file.** It is listed in `.gitignore`.

---

## ▶️ Running SENTINEL

```bash
# Terminal 1 — Start LLaMA-3 server
ollama serve

# Terminal 2 — Launch Streamlit demo
cd SENTINEL
conda activate sentinel_env
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🗂️ Project Structure

```
SENTINEL/
├── pipeline.py              ← End-to-end orchestration
├── app.py                   ← Streamlit demo UI
├── .env                     ← API keys (not committed)
├── requirements.txt
├── modules/
│   ├── module1/             ← Multimodal Encoder
│   │   ├── encoder.py
│   │   └── fusion.py
│   ├── module2/             ← RAG Claim Verifier
│   │   ├── module2.py
│   │   ├── retriever.py     ← 3-source live retrieval
│   │   └── nli_judge.py
│   ├── module3/             ← Heterogeneous GNN
│   │   ├── module3.py
│   │   └── graph_builder.py
│   └── module4/             ← Constitutional Adjudicator
│       └── module4.py
```

---

## 📊 Target Evaluation (Planned — Post Training)

| Configuration | Target F1 |
|---|---|
| Full SENTINEL (all 4 modules) | ~0.91 |
| Without Module 1 (text only) | ~0.87 |
| Without Module 2 (no RAG) | ~0.88 |
| Without Module 3 (no GNN) | ~0.87 |
| Without Module 4 (single-pass) | ~0.89 |
| RoBERTa baseline only | ~0.84 |

> These are projected targets based on architecture design and published benchmarks — not yet empirically measured. Ablation studies are planned after FakeNewsNet training.

---

## 🧪 Datasets (Planned for Training Phase)

| Dataset | Size | Purpose |
|---|---|---|
| FakeNewsNet (PolitiFact) | 23,196 articles | Primary training + evaluation |
| FakeNewsNet (GossipCop) | — | Cross-domain generalization test |
| VERITE | 1,000 pairs | Module 1 pre-training |
| LIAR-PLUS | 12,836 statements | Module 2 NLI fine-tuning |
| NewsCLIPpings | 71,000 image-captions | CLIP branch training |
| Wikipedia Dump | 21M paragraphs | Live retrieval corpus |

---

## 🔭 What's Next

- [ ] Fine-tune on FakeNewsNet (PolitiFact split)
- [ ] Run ablation studies — measure each module's contribution
- [ ] Cross-domain evaluation (PolitiFact → GossipCop)
- [ ] SHAP visualizations for verdict explainability
- [ ] RLHF layer on the adjudicator

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language Models | RoBERTa-base, RoBERTa-large, LLaMA-3 8B |
| Vision | CLIP ViT-B/32 (OpenCLIP) |
| Graph ML | PyTorch Geometric 2.7.0, HGT |
| Retrieval | DPR, FAISS |
| NLP | SpaCy, HuggingFace Transformers |
| Live APIs | Tavily Search, Google Fact Check, Wikipedia Live |
| LLM Runtime | Ollama (4-bit quantized) |
| UI | Streamlit |
| Environment | Python 3.10, conda, Mac mini M4 |

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgements

- Anthropic — Constitutional AI methodology (Bai et al., 2022)
- Meta AI — DPR (Dense Passage Retrieval) and LLaMA-3
- OpenAI / OpenCLIP — CLIP ViT-B/32
- HuggingFace — Transformers, RoBERTa
- PyTorch Geometric — HGT implementation

---

*SENTINEL is an ongoing research and engineering project. Contributions, feedback, and ideas are welcome.*
