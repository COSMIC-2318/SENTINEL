# modules/module1_multimodal/fusion.py

import torch
import torch.nn as nn
from pathlib import Path

class CrossModalAttention(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, hidden_dim=256):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project both branches to same size (hidden_dim)
        self.text_proj  = nn.Linear(text_dim, hidden_dim)   # 768 → 256
        self.image_proj = nn.Linear(image_dim, hidden_dim)  # 512 → 256

        # Q comes from text  — "what is the text looking for?"
        # K comes from image — "what does the image contain?"
        # V comes from image — "what information does the image give?"
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

        # Final classification head: 256 → 2 (Real or Fake)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_vector: torch.Tensor, image_vector: torch.Tensor):
        """
        text_vector:  [1, 768] from RoBERTa
        image_vector: [1, 512] from CLIP
        Returns: output [1, 2], attention_score [1, 1]
        """
        # Step 1 — Project both to hidden_dim
        text_proj  = self.text_proj(text_vector)    # [1, 256]
        image_proj = self.image_proj(image_vector)  # [1, 256]

        # Step 2 — Compute Q, K, V
        Q = self.Q(text_proj)    # text asking the question
        K = self.K(image_proj)   # image saying what it contains
        V = self.V(image_proj)   # image giving its information

        # Step 3 — Attention score (how relevant is image to text?)
        scale = self.hidden_dim ** 0.5
        attention_score = torch.softmax((Q @ K.T) / scale, dim=-1)

        # Step 4 — Weighted image information
        attended = attention_score * V

        # Step 5 — Fuse: text enriched with visual context
        fused = text_proj + attended  # [1, 256]

        # Step 6 — Classify
        output = self.classifier(fused)  # [1, 2]

        return output, attention_score, fused


class PretrainedCrossModal(nn.Module):
    """
    Mirrors the architecture from training/pretrain_module1.py.
    Loads weights from models/cross_modal_pretrained/cross_modal_attention.pt.

    Input:  CLIP text [B,512] + CLIP image [B,512]
    Output: mismatch logit [B,1]  (sigmoid → probability)
    Trained on NewsCLIPpings — detects image-text consistency mismatches.
    """
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, text_emb: torch.Tensor,
                image_emb: torch.Tensor) -> torch.Tensor:
        text_q  = text_emb.unsqueeze(1)
        image_k = image_emb.unsqueeze(1)
        image_v = image_emb.unsqueeze(1)
        attended, _ = self.cross_attention(text_q, image_k, image_v)
        attended    = self.norm1(attended + text_q)
        ffn_out     = self.ffn(attended)
        ffn_out     = self.norm2(ffn_out + attended)
        fused       = ffn_out.squeeze(1)
        combined    = torch.cat([fused, image_emb], dim=1)
        return self.classifier(combined)


def load_pretrained_cross_modal() -> PretrainedCrossModal:
    """
    Loads the pretrained cross-modal attention weights.
    Returns None if weights file is not found.
    """
    weights_path = (
        Path(__file__).resolve().parent.parent.parent
        / "models" / "cross_modal_pretrained" / "cross_modal_attention.pt"
    )
    if not weights_path.exists():
        print(f"  [Module1] Pretrained cross-modal weights not found: {weights_path}")
        return None

    model = PretrainedCrossModal()
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("  [Module1] Pretrained cross-modal attention loaded.")
    return model


# Quick test
if __name__ == "__main__":
    model = CrossModalAttention()

    # Fake vectors to test shapes
    text_vec  = torch.randn(1, 768)
    image_vec = torch.randn(1, 512)

    output, attention = model(text_vec, image_vec)
    probs = torch.softmax(output, dim=-1)

    print(f"Output shape: {output.shape}")
    print(f"P(Real): {probs[0][0].item():.4f}")
    print(f"P(Fake): {probs[0][1].item():.4f}")