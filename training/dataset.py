"""
================================================================
SENTINEL — Dataset Loader
File: training/dataset.py

What this file does:
    Walks through FakeNewsNet article folders, reads each
    news content.json, downloads the top image, and returns
    (text, image, label) ready for the training loop.

How PyTorch datasets work:
    PyTorch expects a Dataset class with two methods:
    - __len__()     → returns total number of samples
    - __getitem__() → returns one sample given an index

    The DataLoader then wraps this class and automatically:
    - Batches samples together (e.g. 32 at a time)
    - Shuffles data each epoch
    - Loads data in parallel using multiple CPU workers

Usage:
    from training.dataset import FakeNewsDataset
    from torch.utils.data import DataLoader

    dataset = FakeNewsDataset(data_root="data/fakenewsnet/...")
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)
================================================================
"""

import os
import json
import requests
import torch
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ── Image transformation pipeline ────────────────────────────
# Every image fed to CLIP must be:
# - Resized to 224x224 pixels (what ViT-B/32 expects)
# - Converted to RGB (some images are grayscale or RGBA)
# - Normalized with CLIP's specific mean and std values
#   (these come from the ImageNet distribution CLIP was trained on)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],  # CLIP mean
        std= [0.26862954, 0.26130258, 0.27577711]   # CLIP std
    )
])

# ── Blank image fallback ──────────────────────────────────────
# When an article has no image or the image URL is dead,
# we use a blank grey image instead of skipping the article.
# This is important — we don't want to lose the text signal
# just because an image is unavailable.
BLANK_IMAGE = IMAGE_TRANSFORM(Image.fromarray(
    np.ones((224, 224, 3), dtype=np.uint8) * 128  # grey
))


def load_image_from_url(url: str, timeout: int = 5) -> torch.Tensor:
    """
    Downloads an image from a URL and transforms it.
    Returns BLANK_IMAGE if download fails or URL is dead.

    Why timeout=5?
        Training iterates thousands of times. If one image URL
        hangs for 30 seconds, training slows to a crawl.
        5 seconds is generous enough for most URLs.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return IMAGE_TRANSFORM(image)
    except Exception:
        return BLANK_IMAGE


def load_image_from_path(path: str) -> torch.Tensor:
    """
    Loads an image from local disk and transforms it.
    Returns BLANK_IMAGE if file is missing or corrupted.
    """
    try:
        image = Image.open(path).convert("RGB")
        return IMAGE_TRANSFORM(image)
    except Exception:
        return BLANK_IMAGE


class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset for FakeNewsNet articles.

    Folder structure it expects:
        data_root/
            politifact/
                fake/
                    politifact11773/
                        news content.json
                        top_image.jpg  (if available)
                real/
                    politifact12345/
                        news content.json
            gossipcop/    (optional)
                fake/
                    ...
                real/
                    ...

    Each sample returned is a dict:
        {
            "text"  : str           ← article body text
            "image" : torch.Tensor  ← [3, 224, 224] image tensor
            "label" : torch.Tensor  ← 1 = fake, 0 = real
            "title" : str           ← article headline
            "source": str           ← politifact or gossipcop
        }
    """

    def __init__(
        self,
        data_root       : str,
        sources         : list = ["politifact"],
        min_text_length : int  = 50,
        use_local_images: bool = True,
        verbose         : bool = True
    ):
        """
        Args:
            data_root        : path to fakenewsnet_dataset folder
            sources          : which news sources to include
                               ["politifact"] or ["politifact", "gossipcop"]
            min_text_length  : skip articles with fewer than this many
                               characters in their text field.
                               Why 50? Anything shorter is likely a
                               scraped error, not a real article.
            use_local_images : if True, look for saved image files on disk
                               before trying to download from URL
            verbose          : print loading progress
        """
        self.data_root        = Path(data_root)
        self.min_text_length  = min_text_length
        self.use_local_images = use_local_images

        # This list holds all valid samples as dicts
        # Each dict has: text, image_source, label, title, source
        self.samples = []

        # ── Walk through all source/label combinations ────────
        for source in sources:
            for label_name, label_value in [("fake", 1), ("real", 0)]:
                folder = self.data_root / source / label_name

                if not folder.exists():
                    if verbose:
                        print(f"  ⚠️  Folder not found: {folder}")
                    continue

                article_dirs = sorted(folder.glob("*"))
                loaded = 0
                skipped_empty = 0
                skipped_short = 0

                for article_dir in article_dirs:
                    json_path = article_dir / "news content.json"

                    # Skip if JSON doesn't exist
                    if not json_path.exists():
                        skipped_empty += 1
                        continue

                    # Load the JSON
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception:
                        skipped_empty += 1
                        continue

                    # Extract text — skip if too short
                    # Why? Short text = scraping failed or Facebook post
                    text = data.get("text", "").strip()
                    if len(text) < min_text_length:
                        skipped_short += 1
                        continue

                    # Get title (use empty string if missing)
                    title = data.get("title", "").strip()

                    # Combine title + text for richer input
                    # Why? The headline often contains the most
                    # sensationalist language in fake news articles
                    full_text = f"{title}. {text}" if title else text

                    # Truncate to 512 tokens worth of characters
                    # RoBERTa has a 512 token limit
                    full_text = full_text[:2000]

                    # Find image — local file first, then URL
                    image_source = None

                    if use_local_images:
                        # Check for locally saved image files
                        for ext in ["top_image.jpg", "top_image.png",
                                    "top_img.jpg", "image.jpg"]:
                            img_path = article_dir / ext
                            if img_path.exists():
                                image_source = str(img_path)
                                break

                    # Fall back to URL if no local image found
                    if image_source is None:
                        top_img = data.get("top_img", "")
                        if top_img and top_img.startswith("http"):
                            image_source = top_img  # URL string

                    # Add to samples list
                    self.samples.append({
                        "text"        : full_text,
                        "image_source": image_source,  # path or URL or None
                        "label"       : label_value,
                        "title"       : title,
                        "source"      : source,
                        "article_id"  : article_dir.name
                    })
                    loaded += 1

                if verbose:
                    print(f"  ✅ {source}/{label_name}: "
                          f"{loaded} loaded, "
                          f"{skipped_short} too short, "
                          f"{skipped_empty} empty")

        if verbose:
            print(f"\n  Total samples: {len(self.samples)}")
            fake_count = sum(1 for s in self.samples if s["label"] == 1)
            real_count = sum(1 for s in self.samples if s["label"] == 0)
            print(f"  Fake: {fake_count}  |  Real: {real_count}")

    def __len__(self) -> int:
        """Returns total number of valid samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns one sample given its index.

        This is called automatically by PyTorch's DataLoader
        when building each batch. DataLoader calls this 32 times
        (for batch_size=32) and stacks the results into tensors.
        """
        sample = self.samples[idx]

        # ── Load image ────────────────────────────────────────
        image_source = sample["image_source"]

        if image_source is None:
            # No image available — use blank grey image
            image = BLANK_IMAGE

        elif image_source.startswith("http"):
            # Download from URL
            image = load_image_from_url(image_source)

        else:
            # Load from local disk
            image = load_image_from_path(image_source)

        return {
            "text"      : sample["text"],
            "image"     : image,                          # [3, 224, 224]
            "label"     : torch.tensor(sample["label"],
                                       dtype=torch.float32),
            "title"     : sample["title"],
            "source"    : sample["source"],
            "article_id": sample["article_id"]
        }


def create_data_splits(
    data_root   : str,
    sources     : list = ["politifact"],
    train_ratio : float = 0.8,
    val_ratio   : float = 0.1,
    test_ratio  : float = 0.1,
    batch_size  : int   = 16,
    num_workers : int   = 0,
    seed        : int   = 42
):
    """
    Creates train / validation / test DataLoaders from FakeNewsNet.

    Why batch_size=16 default?
        M4 Mac mini has limited VRAM. 16 is safe.
        Increase to 32 if training runs without memory errors.

    Why num_workers=0 default?
        On Mac, multiprocessing with PyTorch can cause issues.
        0 means data loading happens in the main process.
        Safe default — increase to 2-4 if training is slow.

    Why seed=42?
        Seeds the random split so every run produces the same
        train/val/test split. Without this, you might accidentally
        train on articles you later test on across runs.

    Returns:
        train_loader, val_loader, test_loader, dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    # Load full dataset
    dataset = FakeNewsDataset(
        data_root=data_root,
        sources=sources,
        verbose=True
    )

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size   = int(total * val_ratio)
    test_size  = total - train_size - val_size  # remainder goes to test

    print(f"\n  Split sizes:")
    print(f"  Train      : {train_size} samples")
    print(f"  Validation : {val_size} samples")
    print(f"  Test       : {test_size} samples")

    # Random split — uses seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Create DataLoaders
    # shuffle=True for train — different order each epoch
    # shuffle=False for val/test — order doesn't matter for evaluation
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn  # handles variable length text
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, dataset


def collate_fn(batch: list) -> dict:
    """
    Custom collate function for DataLoader.

    Why do we need this?
        By default PyTorch tries to stack all fields into tensors.
        But "text", "title", "source", "article_id" are strings —
        you can't stack strings into a tensor.

        This function manually handles each field:
        - Strings → kept as lists
        - Tensors → stacked normally
    """
    return {
        "text"      : [item["text"]       for item in batch],
        "image"     : torch.stack([item["image"] for item in batch]),
        "label"     : torch.stack([item["label"] for item in batch]),
        "title"     : [item["title"]      for item in batch],
        "source"    : [item["source"]     for item in batch],
        "article_id": [item["article_id"] for item in batch],
    }


# ================================================================
# QUICK TEST — run this file directly to verify dataset loads
# Usage: python training/dataset.py
# ================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Find the data root
    SENTINEL_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = (
        SENTINEL_ROOT
        / "data/fakenewsnet/FakeNewsNet/code/fakenewsnet_dataset"
    )

    print("=" * 60)
    print("SENTINEL — Dataset Loader Test")
    print(f"Looking for data at: {DATA_ROOT}")
    print("=" * 60)

    if not DATA_ROOT.exists():
        print(f"❌ Data root not found: {DATA_ROOT}")
        sys.exit(1)

    # Create splits
    train_loader, val_loader, test_loader, dataset = create_data_splits(
        data_root  = str(DATA_ROOT),
        sources    = ["politifact"],
        batch_size = 4,   # small batch just for testing
    )

    print("\n  Testing DataLoader — loading first batch...")
    batch = next(iter(train_loader))

    print(f"\n  ✅ First batch loaded successfully!")
    print(f"  Batch keys   : {list(batch.keys())}")
    print(f"  Image shape  : {batch['image'].shape}")   # [4, 3, 224, 224]
    print(f"  Label shape  : {batch['label'].shape}")   # [4]
    print(f"  Labels       : {batch['label'].tolist()}")
    print(f"  Text[0][:100]: {batch['text'][0][:100]}...")
    print(f"  Title[0]     : {batch['title'][0]}")
    print(f"  Source[0]    : {batch['source'][0]}")

    print("\n  ✅ dataset.py is working correctly!")
    print("  Next step: python training/train.py")
    print("=" * 60)
    
