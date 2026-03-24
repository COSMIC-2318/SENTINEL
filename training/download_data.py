"""
================================================================
SENTINEL — Master Dataset Downloader (v2 — Fixed)
File: training/download_data.py

Run this once. Downloads all 4 datasets into data/ folder.

Usage:
    cd ~/Ankit/SENTINEL
    conda activate sentinel_env
    python training/download_data.py

What gets downloaded:
    data/newsclippings/   ← NewsCLIPpings  (Module 1 pre-training)
    data/verite/          ← VERITE          (Module 1 fine-tuning)
    data/liar_plus/       ← LIAR-PLUS       (Module 2 NLI fine-tuning)
    data/fakenewsnet/     ← FakeNewsNet     (Joint end-to-end training)
================================================================
"""

import os
import sys
import json
import zipfile
import subprocess
from pathlib import Path

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

# ── Root paths ────────────────────────────────────────────────
SENTINEL_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = SENTINEL_ROOT / "data"

for folder in ["newsclippings", "verite", "liar_plus", "fakenewsnet"]:
    (DATA_ROOT / folder).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SENTINEL — Master Dataset Downloader v2")
print(f"Saving all data to: {DATA_ROOT}")
print("=" * 60)


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def download_file(url, dest_path, description=""):
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"  ✅ Already exists, skipping: {dest_path.name}")
        return True
    print(f"  ⬇️  Downloading {description or dest_path.name} ...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
        print(f"\n  ✅ Saved to: {dest_path}")
        return True
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        return False


# ================================================================
# DATASET 1 — NewsCLIPpings
# Source  : GitHub release ZIP (NOT HuggingFace)
# URL     : github.com/g-luo/news_clippings
# Size    : ~71K image-caption pairs (annotations only, ~50MB)
#
# WHAT THIS CONTAINS:
#   JSON files with image URLs + captions + mismatch labels.
#   Images are NOT included in the ZIP — they are linked by URL
#   and downloaded separately during training.
#   Label: 1 = mismatched (fake), 0 = matched (real)
#
# WHY THE HuggingFace ID FAILED:
#   The dataset was never uploaded to HuggingFace under that name.
#   The authors only publish it as a GitHub release ZIP.
# ================================================================
section("DATASET 1 — NewsCLIPpings (Module 1 Pre-training)")

nc_path = DATA_ROOT / "newsclippings"
nc_zip  = nc_path / "news_clippings.zip"
nc_done = nc_path / "summary.json"

if nc_done.exists():
    print("  ✅ NewsCLIPpings already downloaded.")
else:
    NC_URL = (
        "https://github.com/g-luo/news_clippings/releases/download/"
        "v1.0/news_clippings.zip"
    )
    success = download_file(NC_URL, nc_zip, "NewsCLIPpings annotations")

    if not success:
        print()
        print("  ⚠️  Automatic download failed. Manual steps:")
        print("  1. Go to: https://github.com/g-luo/news_clippings/releases")
        print("  2. Download: news_clippings.zip")
        print(f"  3. Place at: {nc_zip}")
        print("  4. Re-run this script.")
    else:
        print("  📦 Extracting...")
        with zipfile.ZipFile(nc_zip, "r") as zf:
            zf.extractall(nc_path)

        # Count samples from extracted JSON files
        train_count = val_count = test_count = 0
        for jf in nc_path.rglob("*.json"):
            with open(jf) as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        name = jf.stem.lower()
                        if "train" in name:
                            train_count = len(data)
                        elif "val" in name:
                            val_count = len(data)
                        elif "test" in name:
                            test_count = len(data)
                except:
                    pass

        summary = {
            "train_size": train_count,
            "val_size"  : val_count,
            "test_size" : test_count,
            "source"    : "github.com/g-luo/news_clippings"
        }
        with open(nc_done, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  ✅ NewsCLIPpings ready.")
        print(f"     Train : {train_count:,}")
        print(f"     Val   : {val_count:,}")
        print(f"     Test  : {test_count:,}")


# ================================================================
# DATASET 2 — VERITE
# Source  : HuggingFace correct ID = stevejpapad/verite
#           Fallback: GitHub CSV if HuggingFace fails
# Size    : ~1,000 verified image-text manipulation pairs
#
# WHAT THIS CONTAINS:
#   Image-caption pairs with 3 label types:
#   0 = true  (image matches caption correctly)
#   1 = miscaptioned  (real image, wrong caption)
#   2 = out-of-context  (real image, used in wrong context)
#
# WHY THE PREVIOUS ID FAILED:
#   "Factiverse/VERITE" was wrong. The correct author namespace
#   on HuggingFace is "stevejpapad" (the paper's first author).
# ================================================================
section("DATASET 2 — VERITE (Module 1 Fine-tuning)")

verite_path = DATA_ROOT / "verite"
verite_done = verite_path / "summary.json"

if verite_done.exists():
    print("  ✅ VERITE already downloaded.")
else:
    try:
        from datasets import load_dataset
        print("  Loading VERITE from HuggingFace (stevejpapad/verite)...")
        dataset = load_dataset("stevejpapad/verite", cache_dir=str(verite_path))

        summary = {
            "splits"  : list(dataset.keys()),
            "sizes"   : {k: len(v) for k, v in dataset.items()},
            "source"  : "huggingface.co/datasets/stevejpapad/verite"
        }
        with open(verite_done, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  ✅ VERITE downloaded.")
        for split, size in summary["sizes"].items():
            print(f"     {split}: {size:,} samples")

        sample = list(dataset.values())[0][0]
        print(f"  Sample keys: {list(sample.keys())}")

    except Exception as e:
        print(f"  ❌ HuggingFace failed: {e}")
        print("  Trying GitHub CSV fallback...")

        VERITE_URL = (
            "https://raw.githubusercontent.com/stevejpapad/"
            "outfox-unlearn/main/data/VERITE/VERITE.csv"
        )
        csv_path = verite_path / "VERITE.csv"
        success = download_file(VERITE_URL, csv_path, "VERITE.csv")

        if success:
            import csv
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            print(f"  ✅ VERITE downloaded via GitHub fallback.")
            print(f"     Total samples : {len(rows):,}")
            print(f"     Columns       : {list(rows[0].keys()) if rows else 'unknown'}")
            summary = {
                "total"  : len(rows),
                "source" : "github fallback",
                "columns": list(rows[0].keys()) if rows else []
            }
            with open(verite_done, "w") as f:
                json.dump(summary, f, indent=2)
        else:
            print()
            print("  ⚠️  Manual download required:")
            print("  1. Go to: https://github.com/stevejpapad/outfox-unlearn")
            print("  2. Download VERITE.csv from data/VERITE/")
            print(f"  3. Place at: {verite_path}/VERITE.csv")
            print("  4. Re-run this script.")


# ================================================================
# DATASET 3 — LIAR-PLUS  (already downloaded ✅)
# ================================================================
section("DATASET 3 — LIAR-PLUS (Module 2 NLI Fine-tuning)")

liar_path  = DATA_ROOT / "liar_plus"
liar_files = ["train.tsv", "val.tsv", "test.tsv"]

if all((liar_path / f).exists() for f in liar_files):
    for fname in liar_files:
        with open(liar_path / fname) as f:
            lines = sum(1 for _ in f)
        print(f"  ✅ {fname}: {lines:,} statements")
else:
    LIAR_URLS = {
        "train.tsv": "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/train2.tsv",
        "val.tsv"  : "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/val2.tsv",
        "test.tsv" : "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/test2.tsv",
    }
    for fname, url in LIAR_URLS.items():
        download_file(url, liar_path / fname, f"LIAR-PLUS {fname}")


# ================================================================
# DATASET 4 — FakeNewsNet
# Source  : GitHub clone → run their download_manager.py script
# URL     : github.com/KaiDMML/FakeNewsNet
#
# WHY pip install FAILED:
#   FakeNewsNet is a research tool, not a pip package.
#   It exists only as a Python script on GitHub.
#   You clone the repo, then run their downloader which crawls
#   live news websites and saves articles to your disk.
#
# TIME WARNING: Crawl takes 30–60 minutes.
#   Expected yield: ~5,000–8,000 articles (60–70% of total,
#   since many URLs from 2016–2020 are now dead).
# ================================================================
section("DATASET 4 — FakeNewsNet (Joint Training)")

fnn_path = DATA_ROOT / "fakenewsnet"
fnn_repo = fnn_path / "FakeNewsNet"

if fnn_repo.exists():
    print("  ✅ FakeNewsNet repository already cloned.")
else:
    print("  Cloning FakeNewsNet from GitHub...")
    result = subprocess.run(
        ["git", "clone",
         "https://github.com/KaiDMML/FakeNewsNet.git",
         str(fnn_repo)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ✅ Repository cloned successfully.")
    else:
        print(f"  ❌ Git clone failed.")
        print(f"     Error: {result.stderr.strip()}")
        print()
        print("  Manual step:")
        print(f"  git clone https://github.com/KaiDMML/FakeNewsNet {fnn_repo}")

# Check if articles already crawled
fnn_data = fnn_repo / "code" / "fakenewsnet_data" / "politifact"
if fnn_data.exists():
    fake_count = len(list((fnn_data / "fake").glob("*"))) if (fnn_data / "fake").exists() else 0
    real_count = len(list((fnn_data / "real").glob("*"))) if (fnn_data / "real").exists() else 0
    if fake_count > 100:
        print(f"  ✅ Articles already crawled.")
        print(f"     PolitiFact fake : {fake_count:,}")
        print(f"     PolitiFact real : {real_count:,}")
    else:
        print("  ⚠️  Repo cloned but articles not yet crawled.")

print()
print("  ─── Run this separately to crawl the articles (30–60 min) ───")
print()
print(f"  cd {fnn_repo}/code")
print( "  python download_manager.py \\")
print( "      --news_source politifact \\")
print( "      --data_type fake real \\")
print( "      --article True \\")
print( "      --social_context False \\")
print( "      --top_image True")
print()

sh_path = fnn_path / "crawl_fakenewsnet.sh"
with open(sh_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"cd {fnn_repo}/code\n")
    f.write("python download_manager.py \\\n")
    f.write("    --news_source politifact \\\n")
    f.write("    --data_type fake real \\\n")
    f.write("    --article True \\\n")
    f.write("    --social_context False \\\n")
    f.write("    --top_image True\n")
os.chmod(sh_path, 0o755)
print(f"  ✅ Saved to: {sh_path}")
print(f"     Shortcut : bash data/fakenewsnet/crawl_fakenewsnet.sh")


# ================================================================
# FINAL SUMMARY
# ================================================================
section("DOWNLOAD SUMMARY")

checks = {
    "NewsCLIPpings" : (DATA_ROOT / "newsclippings" / "summary.json").exists(),
    "VERITE"        : (DATA_ROOT / "verite" / "summary.json").exists(),
    "LIAR-PLUS"     : (DATA_ROOT / "liar_plus" / "train.tsv").exists(),
    "FakeNewsNet"   : (DATA_ROOT / "fakenewsnet" / "FakeNewsNet").exists(),
}

for name, done in checks.items():
    status = "✅ Ready  " if done else "⏳ Pending"
    print(f"  {status}  {name}")

all_ready = all(checks.values())
print()
if all_ready:
    print("  🎉 All datasets ready!")
    print("  Next: python training/phase1_train.py")
else:
    print("  Next Steps:")
    print("  1. Run FakeNewsNet crawl (30–60 min):")
    print("     bash data/fakenewsnet/crawl_fakenewsnet.sh")
    print("  2. Once all 4 show ✅ → python training/phase1_train.py")
print()
print("=" * 60)