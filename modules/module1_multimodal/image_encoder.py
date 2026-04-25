# modules/module1_multimodal/image_encoder.py
# SENTINEL — Image Branch
# Converts article image into a 512-dimensional meaning vector using CLIP

import open_clip
import torch
from PIL import Image

class ImageEncoder:
    def __init__(self):
        # Load pre-trained CLIP ViT-B/32
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        self.model.eval()  # inference mode — no training yet

    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        Takes a PIL Image
        Returns a [1, 512] tensor — the image meaning vector
        """
        image_input = self.preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            image_vector = self.model.encode_image(image_input)  # [1, 512]
        return image_vector

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encodes text using CLIP's text encoder.
        Returns a [1, 512] tensor in the same embedding space as encode().
        Used by the pretrained cross-modal attention layer.
        """
        tokenizer   = open_clip.get_tokenizer('ViT-B-32')
        text_tokens = tokenizer([text])
        with torch.no_grad():
            text_vector = self.model.encode_text(text_tokens)  # [1, 512]
        return text_vector.float()


# Quick test — run this file directly to verify it works
if __name__ == "__main__":
    import numpy as np

    # Create a test image
    encoder = ImageEncoder()
    test_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    vector = encoder.encode(test_image)
    print(f"Output vector shape: {vector.shape}")
    print(f"First 5 values: {vector[0][:5]}")