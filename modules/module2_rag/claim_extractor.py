import spacy

nlp = spacy.load("en_core_web_sm")

ASSERTIVE_VERBS = {
    "confirm", "announce", "sign", "launch", "kill", "discover",
    "arrest", "declare", "release", "publish", "reveal", "report",
    "state", "claim", "say", "warn", "approve", "reject", "ban",
    "win", "won", "lose", "lost", "defeat", "beat", "elect", "vote",
    "lead", "gain", "secure", "capture", "sweep", "take", "hold",
    "retain", "fail", "fall", "drop", "rise", "increase", "decrease",
    "pass", "block", "file", "accuse", "charge", "convict",
    "resign", "appoint", "suspend", "fire", "hire", "merge",
    "die", "kill", "attack", "strike", "protest", "rally", "summit"
}

def extract_claims(article_text):
    """
    Input  — full article text as a string
    Output — list of sentences that are checkable factual claims
    """
    doc = nlp(article_text)
    claims = []

    for sentence in doc.sents:
        has_entity = len(sentence.ents) > 0
        sentence_lemmas = [token.lemma_.lower() for token in sentence]
        has_assertive_verb = any(verb in sentence_lemmas for verb in ASSERTIVE_VERBS)

        if has_entity and has_assertive_verb:
            claims.append(sentence.text.strip())

    # Fallback for short articles or all-caps text where NER fails:
    # if nothing extracted, treat the whole text as one claim
    if not claims:
        stripped = article_text.strip()
        if stripped:
            claims.append(stripped[:300])

    return claims


# --- Test ---
if __name__ == "__main__":
    test_article = """
    The situation is extremely dangerous and everyone is worried.
    NASA confirmed the discovery of water on Mars on March 12, 2026.
    It is a very bad thing that this happened.
    Prime Minister Modi announced a new economic policy in New Delhi yesterday.
    People are feeling scared and uncertain about the future.
    """

    claims = extract_claims(test_article)

    print(f"Found {len(claims)} claims:\n")
    for i, claim in enumerate(claims, 1):
        print(f"Claim {i}: {claim}")