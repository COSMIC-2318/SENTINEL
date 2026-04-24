"""
================================================================
SENTINEL — Dataset Loader
File: training/dataset.py

Supports FakeNewsNet + WELFake combined training.
WELFake samples use a blank grey image as placeholder.
================================================================
"""

import os
import json
import requests
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms

# ── Image transformation pipeline ────────────────────────────
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711]
    )
])

BLANK_IMAGE = IMAGE_TRANSFORM(Image.fromarray(
    np.ones((224, 224, 3), dtype=np.uint8) * 128
))


def load_image_from_url(url: str, timeout: int = 5) -> torch.Tensor:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return IMAGE_TRANSFORM(image)
    except Exception:
        return BLANK_IMAGE


def load_image_from_path(path: str) -> torch.Tensor:
    try:
        image = Image.open(path).convert("RGB")
        return IMAGE_TRANSFORM(image)
    except Exception:
        return BLANK_IMAGE


# ================================================================
# FAKENEWSNET DATASET (unchanged)
# ================================================================
class FakeNewsDataset(Dataset):
    def __init__(
        self,
        data_root       : str,
        sources         : list = ["politifact", "gossipcop"],
        min_text_length : int  = 30,
        use_local_images: bool = True,
        verbose         : bool = True
    ):
        self.data_root        = Path(data_root)
        self.min_text_length  = min_text_length
        self.use_local_images = use_local_images
        self.samples = []

        for source in sources:
            for label_name, label_value in [("fake", 1), ("real", 0)]:
                folder = self.data_root / source / label_name

                if not folder.exists():
                    if verbose:
                        print(f"  ⚠️  Folder not found: {folder}")
                    continue

                article_dirs  = sorted(folder.glob("*"))
                loaded        = 0
                skipped_empty = 0
                skipped_short = 0

                for article_dir in article_dirs:
                    json_path = article_dir / "news content.json"
                    if not json_path.exists():
                        skipped_empty += 1
                        continue
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception:
                        skipped_empty += 1
                        continue

                    text = data.get("text", "").strip()
                    if len(text) < min_text_length:
                        skipped_short += 1
                        continue

                    title     = data.get("title", "").strip()
                    full_text = f"{title}. {text}" if title else text
                    full_text = full_text[:2000]

                    image_source = None
                    if use_local_images:
                        for ext in ["top_image.jpg", "top_image.png",
                                    "top_img.jpg", "image.jpg"]:
                            img_path = article_dir / ext
                            if img_path.exists():
                                image_source = str(img_path)
                                break

                    if image_source is None:
                        top_img = data.get("top_img", "")
                        if top_img and top_img.startswith("http"):
                            image_source = top_img

                    self.samples.append({
                        "text"        : full_text,
                        "image_source": image_source,
                        "label"       : label_value,
                        "title"       : title,
                        "source"      : source,
                        "article_id"  : article_dir.name
                    })
                    loaded += 1

                if verbose:
                    print(f"  ✅ {source}/{label_name}: "
                          f"{loaded} loaded, {skipped_short} too short, "
                          f"{skipped_empty} empty")

        if verbose:
            fake_c = sum(1 for s in self.samples if s["label"] == 1)
            real_c = sum(1 for s in self.samples if s["label"] == 0)
            print(f"\n  FakeNewsNet total: {len(self.samples)} "
                  f"(Fake: {fake_c} | Real: {real_c})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample       = self.samples[idx]
        image_source = sample["image_source"]

        if image_source is None:
            image = BLANK_IMAGE
        elif image_source.startswith("http"):
            image = load_image_from_url(image_source)
        else:
            image = load_image_from_path(image_source)

        return {
            "text"      : sample["text"],
            "image"     : image,
            "label"     : torch.tensor(sample["label"], dtype=torch.float32),
            "title"     : sample["title"],
            "source"    : sample["source"],
            "article_id": sample["article_id"]
        }


# ================================================================
# WELFAKE DATASET — NEW
# ================================================================
class WELFakeDataset(Dataset):
    """
    WELFake: 72,134 articles (35K fake + 37K real), text only.
    CSV columns: Serial, Title, Text, label  (0=Real, 1=Fake)

    No images — uses BLANK_IMAGE as placeholder so the pipeline
    doesn't need any architectural changes.
    """

    def __init__(
        self,
        csv_path        : str,
        min_text_length : int = 30,
        max_samples     : int = None,
        verbose         : bool = True
    ):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text"])
        df = df[df["text"].str.strip().str.len() >= min_text_length]
        df = df[df["label"].isin([0, 1])].reset_index(drop=True)

        # Optional cap — keeps class balance
        if max_samples is not None:
            half    = max_samples // 2
            fake_df = df[df["label"] == 1].head(half)
            real_df = df[df["label"] == 0].head(half)
            df = pd.concat([fake_df, real_df]).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)

        self.df = df

        if verbose:
            fake_c = (df["label"] == 1).sum()
            real_c = (df["label"] == 0).sum()
            print(f"\n  WELFake total: {len(df)} "
                  f"(Fake: {fake_c} | Real: {real_c})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        title = str(row.get("title", "")) if pd.notna(row.get("title", np.nan)) else ""
        text  = str(row["text"]).strip()
        full_text = f"{title}. {text}" if title else text
        full_text = full_text[:2000]

        return {
            "text"      : full_text,
            "image"     : BLANK_IMAGE,
            "label"     : torch.tensor(int(row["label"]), dtype=torch.float32),
            "title"     : title,
            "source"    : "welfake",
            "article_id": str(idx)
        }


# ================================================================
# COLLATE FUNCTION (unchanged)
# ================================================================
def collate_fn(batch: list) -> dict:
    return {
        "text"      : [item["text"]       for item in batch],
        "image"     : torch.stack([item["image"] for item in batch]),
        "label"     : torch.stack([item["label"] for item in batch]),
        "title"     : [item["title"]      for item in batch],
        "source"    : [item["source"]     for item in batch],
        "article_id": [item["article_id"] for item in batch],
    }


# ================================================================
# CREATE DATA SPLITS — now accepts welfake_csv
# ================================================================
def create_data_splits(
    data_root           : str,
    sources             : list  = ["politifact"],
    train_ratio         : float = 0.8,
    val_ratio           : float = 0.1,
    test_ratio          : float = 0.1,
    batch_size          : int   = 16,
    num_workers         : int   = 0,
    seed                : int   = 42,
    welfake_csv         : str   = None,   # ← NEW
    welfake_max_samples : int   = None,   # ← NEW
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    datasets = []

    # FakeNewsNet
    fnn = FakeNewsDataset(data_root=data_root, sources=sources, verbose=True)
    if len(fnn) > 0:
        datasets.append(fnn)

    # WELFake
    if welfake_csv and Path(welfake_csv).exists():
        wf = WELFakeDataset(
            csv_path=welfake_csv,
            max_samples=welfake_max_samples,
            verbose=True
        )
        if len(wf) > 0:
            datasets.append(wf)
    elif welfake_csv:
        print(f"  ⚠️  WELFake CSV not found: {welfake_csv}")

    if not datasets:
        raise ValueError("No data found. Check your paths.")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    total    = len(combined)

    train_size = int(total * train_ratio)
    val_size   = int(total * val_ratio)
    test_size  = total - train_size - val_size

    print(f"\n  Combined: {total} samples")
    print(f"  Train: {train_size} | Val: {val_size} | Test: {test_size}")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        combined, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, combined


# ================================================================
# QUICK TEST
# ================================================================
if __name__ == "__main__":
    SENTINEL_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT     = (SENTINEL_ROOT /
                     "data/fakenewsnet/FakeNewsNet/code/fakenewsnet_dataset")
    WELFAKE_CSV   = SENTINEL_ROOT / "data" / "WELFake_Dataset.csv"

    print("=" * 60)
    print("SENTINEL — Dataset Loader Test (FakeNewsNet + WELFake)")
    print("=" * 60)

    train_loader, val_loader, test_loader, combined = create_data_splits(
        data_root           = str(DATA_ROOT),
        sources             = ["politifact"],
        batch_size          = 4,
        welfake_csv         = str(WELFAKE_CSV),
        welfake_max_samples = 100,  # small cap for quick test
    )

    batch = next(iter(train_loader))
    print(f"\n  ✅ Batch loaded!")
    print(f"  Image : {batch['image'].shape}")
    print(f"  Labels: {batch['label'].tolist()}")
    print(f"  Sources: {batch['source']}")
    print(f"  Text[0][:80]: {batch['text'][0][:80]}")
    print("\n  ✅ dataset.py working correctly!")