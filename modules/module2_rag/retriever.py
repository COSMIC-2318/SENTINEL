# retriever.py — Live 3-source retrieval with DPR relevance ranking
import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
import os
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import torch.nn.functional as F

# Load API keys from .env at SENTINEL root
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_FACT_CHECK_KEY")

# Load DPR model once at import time (used for relevance ranking)
print("Loading DPR encoder...")
dpr_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
dpr_model = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
dpr_model.eval()
print("DPR ready.")


def embed_text(text):
    """Convert text to a DPR vector for relevance scoring."""
    inputs = dpr_tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512
    )
    with torch.no_grad():
        vector = dpr_model(**inputs).pooler_output
    return vector  # shape [1, 768]


def fetch_wikipedia(claim):
    """Query Wikipedia Live API for background knowledge."""
    passages = []
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": claim,
            "format": "json",
            "srlimit": 3
        }
        headers = {
            "User-Agent": "SENTINEL-FakeNewsDetector/1.0 (research project)"
        }
        response = requests.get(url, params=params, headers=headers, timeout=5)
        results = response.json().get("query", {}).get("search", [])
        for r in results:
            snippet = r.get("snippet", "")
            clean = snippet.replace("<span class=\"searchmatch\">", "") \
                           .replace("</span>", "")
            if clean:
                passages.append({"text": clean, "source": "wikipedia"})
    except Exception as e:
        print(f"Wikipedia fetch failed: {e}")
    return passages


def fetch_tavily(claim):
    """Query Tavily for live web search results."""
    passages = []
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(claim, max_results=3)
        for result in response.get("results", []):
            text = result.get("content", "")
            if text:
                passages.append({"text": text, "source": "tavily"})
    except Exception as e:
        print(f"Tavily fetch failed: {e}")
    return passages


def fetch_google_factcheck(claim):
    """Query Google Fact Check Tools API for pre-verified claims."""
    passages = []
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": claim, "key": GOOGLE_KEY, "pageSize": 3}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        for item in data.get("claims", []):
            text = item.get("text", "")
            reviews = item.get("claimReview", [])
            rating = reviews[0].get("textualRating", "") if reviews else ""
            publisher = reviews[0].get("publisher", {}).get("name", "") if reviews else ""
            if text:
                combined = f"{text} — Rated: {rating} by {publisher}"
                passages.append({"text": combined, "source": "google_factcheck"})
    except Exception as e:
        print(f"Google Fact Check fetch failed: {e}")
    return passages


def rank_passages(claim, passages, top_k=5, threshold=0.65):
    """Use DPR to score all passages by relevance to the claim. Return top_k."""
    if not passages:
        return []

    claim_vector = embed_text(claim)  # [1, 768]
    scored = []

    for p in passages:
        passage_vector = embed_text(p["text"])
        score = F.cosine_similarity(claim_vector, passage_vector).item()
        if score >= threshold:
            scored.append({
                "text": p["text"],
                "source": p["source"],
                "relevance_score": round(score, 4)
            })

    # Sort by relevance, return top_k
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored[:top_k]


def retrieve(claim):
    """
    Main function called by module2.py.
    Takes a claim string, returns top 5 ranked passages from 3 live sources.
    """
    print(f"\nRetrieving evidence for: '{claim}'")

    # Step 1 — Fetch from all 3 sources
    wiki_passages     = fetch_wikipedia(claim)
    tavily_passages   = fetch_tavily(claim)
    google_passages   = fetch_google_factcheck(claim)

    print(f"  Wikipedia: {len(wiki_passages)} passages")
    print(f"  Tavily:    {len(tavily_passages)} passages")
    print(f"  Google FC: {len(google_passages)} passages")

    # Step 2 — Merge all passages into one list
    all_passages = wiki_passages + tavily_passages + google_passages

    if not all_passages:
        print("  WARNING: No evidence found from any source.")
        return []

    # Step 3 — DPR ranks by relevance to the claim
    ranked = rank_passages(claim, all_passages)
    print(f"  After DPR ranking: {len(ranked)} passages above threshold")

    return ranked