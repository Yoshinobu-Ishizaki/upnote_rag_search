#!/usr/bin/env python
"""Preprocessing pipeline for UpNote RAG Search.

Run this script before starting the Streamlit app.
For large note collections (25,000+), this may take several hours.

Usage:
    python preprocess.py [--path "D:/backup/UpNote/General Space"]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess UpNote .upnx backup files for search"
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

    # Step 1: .upnx → upnote_text.csv
    _print_step(1, "Creating text dataframe from .upnx backup files")
    _create_dataframe(args.path)

    # Step 2: upnote_text.csv → upnote_text_split.csv (BM25 tokens)
    _print_step(2, "Tokenizing text for BM25 search")
    _tokenize_for_bm25()

    # Step 3: embeddings.npy + faiss.index
    _print_step(3, "Creating sentence embeddings and FAISS index")
    print("This may take a long time for large note collections.\n")
    _create_embeddings()

    _print_done()


def _create_dataframe(path: str | None) -> None:
    from src.parser import get_backup_root, find_latest_upnx, parse_upnx

    backup_root = get_backup_root(path, PROJECT_ROOT)
    upnx = find_latest_upnx(backup_root)
    print(f"Reading: {upnx.name}")
    df = parse_upnx(upnx)
    print(f"Notes: {len(df)} (active)")
    out = DATA_DIR / "upnote_text.csv"
    df.write_csv(out)
    print(f"Saved: {out}")


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


def _tokenize_for_bm25() -> None:
    import polars as pl
    from tqdm import tqdm
    from src.tokenizer import create_tokenizer, tokenize_text, tokens_to_string

    df = pl.read_csv(DATA_DIR / "upnote_text.csv")
    tokenizer = create_tokenizer()
    contents = df["contents"].fill_null("").to_list()
    tokens_list = [
        tokens_to_string(tokenize_text(s, tokenizer))
        for s in tqdm(contents, desc="Tokenizing", unit="note")
    ]
    df.with_columns(pl.Series("tokens", tokens_list)) \
      .select(pl.exclude("contents")) \
      .write_csv(DATA_DIR / "upnote_text_split.csv")


def _create_embeddings() -> None:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    import polars as pl

    df = pl.read_csv(DATA_DIR / "upnote_text.csv")
    ids      = df["id"].to_list()
    updates  = df["update"].to_list()
    contents = df["contents"].fill_null("").to_list()
    n_total  = len(ids)

    # Load cache
    meta_path = DATA_DIR / "embedding_meta.csv"
    npy_path  = DATA_DIR / "embeddings.npy"
    cache: dict = {}          # id -> (update_ts, old_row_idx)
    old_embeddings = None

    if meta_path.exists() and npy_path.exists():
        meta_df = pl.read_csv(meta_path)
        old_embeddings = np.load(npy_path)
        for row_idx, (cid, cup) in enumerate(
            zip(meta_df["id"].to_list(), meta_df["update"].to_list())
        ):
            cache[cid] = (cup, row_idx)
        print(f"Cache loaded: {len(cache)} entries from previous run.")
    else:
        print("No cache found — encoding all notes from scratch.")

    # Load checkpoint (partial results from a previous interrupted run)
    ckpt_path = DATA_DIR / "embedding_checkpoint.npz"
    ckpt: dict = {}  # id -> (update_ts, np.ndarray embedding)
    if ckpt_path.exists():
        data = np.load(ckpt_path, allow_pickle=True)
        ckpt_ids_arr = data["ids"].tolist()
        ckpt_ups_arr = data["updates"].tolist()
        ckpt_embs_arr = data["embeddings"]  # shape (N, dim)
        for cid, cup, cemb in zip(ckpt_ids_arr, ckpt_ups_arr, ckpt_embs_arr):
            ckpt[cid] = (cup, cemb)
        print(f"Checkpoint loaded: {len(ckpt)} partially-encoded notes resumed.")
    else:
        print("No checkpoint found.")

    # Classify notes
    new_indices, new_contents = [], []
    for i, (nid, nup) in enumerate(zip(ids, updates)):
        if nid in cache and cache[nid][0] == nup:
            pass  # reuse from saved cache
        elif nid in ckpt and ckpt[nid][0] == nup:
            pass  # reuse from checkpoint
        else:
            new_indices.append(i)
            new_contents.append(contents[i])

    n_cached = n_total - len(new_indices)
    print(f"  {n_cached} notes reused from cache/checkpoint, {len(new_indices)} notes to encode.")

    # Encode only new notes
    dim = 768
    if new_contents:
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        print(f"Encoding {len(new_contents)} notes...")

        # Seed ckpt accumulator with already-loaded checkpoint data
        ckpt_ids_list  = list(ckpt.keys())
        ckpt_ups_list  = [ckpt[k][0] for k in ckpt_ids_list]
        ckpt_embs_list = [ckpt[k][1] for k in ckpt_ids_list]

        BATCH = 64
        total_batches = (len(new_contents) + BATCH - 1) // BATCH
        for b in range(total_batches):
            batch_contents = new_contents[b * BATCH : (b + 1) * BATCH]
            batch_indices  = new_indices [b * BATCH : (b + 1) * BATCH]
            batch_emb = model.encode(
                batch_contents, normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")
            dim = batch_emb.shape[1]

            for idx, emb in zip(batch_indices, batch_emb):
                nid, nup = ids[idx], updates[idx]
                ckpt[nid] = (nup, emb)
                ckpt_ids_list.append(nid)
                ckpt_ups_list.append(nup)
                ckpt_embs_list.append(emb)

            np.savez(
                ckpt_path,
                ids=ckpt_ids_list,
                updates=ckpt_ups_list,
                embeddings=np.array(ckpt_embs_list, dtype="float32"),
            )
            print(f"  Batch {b + 1}/{total_batches} done — checkpoint saved ({len(ckpt_ids_list)} notes).")

    else:
        if old_embeddings is not None:
            dim = old_embeddings.shape[1]
        print("All notes served from cache/checkpoint — skipping model load.")

    # Assemble in current CSV row order
    final_embeddings = np.empty((n_total, dim), dtype="float32")
    for i, (nid, nup) in enumerate(zip(ids, updates)):
        if nid in cache and cache[nid][0] == nup:
            final_embeddings[i] = old_embeddings[cache[nid][1]]
        else:
            final_embeddings[i] = ckpt[nid][1]

    # Save outputs
    np.save(npy_path, final_embeddings)
    print(f"Saved: {npy_path}")

    index = faiss.IndexFlatIP(dim)
    index.add(final_embeddings)
    out_faiss = DATA_DIR / "faiss.index"
    faiss.write_index(index, str(out_faiss))
    print(f"Saved: {out_faiss}")

    pl.DataFrame({"id": ids, "update": updates}).write_csv(meta_path)
    print(f"Saved: {meta_path}")

    if ckpt_path.exists():
        ckpt_path.unlink()
        print("Checkpoint deleted (run complete).")


if __name__ == "__main__":
    main()
