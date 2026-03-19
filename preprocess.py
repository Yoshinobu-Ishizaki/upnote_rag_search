#!/usr/bin/env python
"""Preprocessing pipeline for UpNote Markdown Manager.

Run this script before starting the Streamlit app.
For large note collections (25,000+), this may take several hours.

Usage:
    python preprocess.py [--path "D:/backup/UpNote/General Space"]
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess UpNote markdown files for search"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to UpNote backup folder (overrides config.ini)",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Skip Steps 1 & 2 (CSV creation) and only build embeddings/FAISS index from existing CSVs",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if args.embeddings_only:
        if not (DATA_DIR / "upnote_text.csv").exists():
            print("Error: data/upnote_text.csv not found. Run without --embeddings-only first.")
            sys.exit(1)
        _print_step(3, "Creating sentence embeddings and FAISS index (embeddings-only mode)")
        print("This may take a long time for large note collections.\n")
        _create_embeddings()
        _print_done()
        return

    # Step 1: Markdown → upnote_text.csv
    _print_step(1, "Creating text dataframe from markdown files")
    cmd1 = [sys.executable, str(PROJECT_ROOT / "script" / "01_df-creation.py")]
    if args.path:
        cmd1 += ["--path", args.path]
    result = subprocess.run(cmd1, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("\nStep 1 failed. Aborting.")
        sys.exit(1)

    # Step 2: upnote_text.csv → upnote_text_split.csv (BM25 tokens)
    _print_step(2, "Tokenizing text for BM25 search")
    cmd2 = [sys.executable, str(PROJECT_ROOT / "script" / "02_splitwords.py")]
    result = subprocess.run(cmd2, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("\nStep 2 failed. Aborting.")
        sys.exit(1)

    # Step 3: embeddings.npy + faiss.index
    _print_step(3, "Creating sentence embeddings and FAISS index")
    print("This may take a long time for large note collections.\n")
    _create_embeddings()

    _print_done()


def _print_done() -> None:
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"\nGenerated files in {DATA_DIR}:")
    for f in sorted(DATA_DIR.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:40s} {size_mb:8.2f} MB")


def _print_step(n: int, description: str) -> None:
    print()
    print("=" * 60)
    print(f"Step {n}: {description}")
    print("=" * 60)


def _create_embeddings() -> None:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    import polars as pl

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    df = pl.read_csv(DATA_DIR / "upnote_text.csv")
    contents = df["contents"].fill_null("").to_list()
    print(f"Encoding {len(contents)} notes...")

    embeddings = model.encode(
        contents,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype("float32")

    out_npy = DATA_DIR / "embeddings.npy"
    np.save(out_npy, embeddings)
    print(f"Saved: {out_npy}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    out_faiss = DATA_DIR / "faiss.index"
    faiss.write_index(index, str(out_faiss))
    print(f"Saved: {out_faiss}")


if __name__ == "__main__":
    main()
