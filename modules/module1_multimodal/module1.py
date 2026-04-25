# modules/module1_multimodal/module1.py
# SENTINEL — Module 1: Full Multimodal Encoder Pipeline
# Combines TextEncoder + ImageEncoder + CrossModalAttention

import torch
from PIL import Image
from text_encoder import TextEncoder
from image_encoder import ImageEncoder
from fusion import CrossModalAttention, load_pretrained_cross_modal


class Module1:
    def __init__(self):
        print("Loading Module 1 — Multimodal Encoder...")

        self.text_encoder       = TextEncoder()
        self.image_encoder      = ImageEncoder()
        self.fusion             = CrossModalAttention()
        self.pretrained_cm      = load_pretrained_cross_modal()  # may be None

        print("Module 1 ready!")

    def predict(self, text: str, image: Image.Image) -> dict:
        """
        Takes a news article headline/text and its image
        Returns prediction dict with probabilities and verdict

        text:  string — article headline or body
        image: PIL Image — article image
        """
        # Step 1 — Encode text
        text_vector = self.text_encoder.encode(text)         # [1, 768]

        # Step 2 — Encode image
        image_vector = self.image_encoder.encode(image)      # [1, 512]

        # Step 3 — Fuse with Cross-Modal Attention
        output, attention_score, fusion_vector = self.fusion(text_vector, image_vector)

        # Step 4 — Convert to probabilities
        probs = torch.softmax(output, dim=-1)
        p_real = probs[0][0].item()
        p_fake = probs[0][1].item()

            # Step 5 — Verdict
        verdict = "FAKE" if p_fake > 0.5 else "REAL"

        # Step 6 — Interpret attention score
        attention_val = attention_score.item()

        if attention_val > 0.75:
            mismatch_description = "Very high — image content strongly inconsistent with article text"
        elif attention_val > 0.55:
            mismatch_description = "Moderate — image shows partial inconsistency with article text"
        elif attention_val > 0.35:
            mismatch_description = "Low — image broadly consistent with article text"
        else:
            mismatch_description = "Very low — image appears to directly match article text"

        # Step 7 — Pretrained cross-modal mismatch score (NewsCLIPpings trained)
        pretrained_mismatch_prob = None
        if self.pretrained_cm is not None:
            try:
                clip_text_vec = self.image_encoder.encode_text(text)   # [1, 512]
                clip_img_vec  = image_vector.float()                    # [1, 512]
                with __import__('torch').no_grad():
                    logit = self.pretrained_cm(clip_text_vec, clip_img_vec)
                pretrained_mismatch_prob = round(
                    __import__('torch').sigmoid(logit).item(), 4
                )
            except Exception as e:
                print(f"  [Module1] Pretrained cross-modal failed: {e}")

        return {
            "verdict":                  verdict,
            "p_real":                   round(p_real, 4),
            "p_fake":                   round(p_fake, 4),
            "attention_score":          round(attention_val, 4),
            "mismatch_description":     mismatch_description,
            "pretrained_mismatch_prob": pretrained_mismatch_prob,
            "fusion_vector":            fusion_vector,
            "text_vector":              text_vector,
            "image_vector":             image_vector,
        }

# Full pipeline test
if __name__ == "__main__":
    import numpy as np

    # Create module
    module1 = Module1()

    # Test input
    headline = "NASA confirms water found on Mars surface"
    test_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    # Run prediction
    result = module1.predict(headline, test_image)

    print()
    print("=" * 50)
    print("SENTINEL — Module 1 Output")
    print("=" * 50)
    print(f"Headline:        {headline}")
    print(f"Verdict:         {result['verdict']}")
    print(f"P(Real):         {result['p_real']}")
    print(f"P(Fake):         {result['p_fake']}")
    print(f"Attention Score: {result['attention_score']}")
    print()
    print("Note: random weights = random prediction")
    print("This becomes meaningful after training on FakeNewsNet!")