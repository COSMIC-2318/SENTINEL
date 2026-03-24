# modules/module1_multimodal/text_encoder.py
# SENTINEL — Text Branch
# Converts article text into a 768-dimensional meaning vector using RoBERTa

from transformers import RobertaTokenizer, RobertaModel
import torch

class TextEncoder:
    def __init__(self):
        # Load pre-trained RoBERTa
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.model.eval()  # inference mode — no training yet

    def encode(self, text: str) -> torch.Tensor:
        """
        Takes a string of text (article headline or body)
        Returns a [1, 768] tensor — the CLS vector (sentence meaning)
        """
        # Step 1 — Tokenize
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )

        # Step 2 — Pass through RoBERTa
        with torch.no_grad():
            output = self.model(**tokens)

        # Step 3 — Extract CLS token (sentence summary)
        cls_vector = output.last_hidden_state[:, 0, :]  # [1, 768]

        return cls_vector


# Quick test — run this file directly to verify it works
if __name__ == "__main__":
    encoder = TextEncoder()
    text = "NASA confirms water found on Mars surface"
    vector = encoder.encode(text)
    print(f"Input: {text}")
    print(f"Output vector shape: {vector.shape}")
    print(f"First 5 values: {vector[0][:5]}")