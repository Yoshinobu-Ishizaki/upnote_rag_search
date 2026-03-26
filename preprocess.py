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
LOCAL_DATA_DIR = DATA_DIR / "local"
GOOGLE_DATA_DIR = DATA_DIR / "google"


def main() -> None:
    from src.config import get_embedding_provider

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
    LOCAL_DATA_DIR.mkdir(exist_ok=True)

    provider = get_embedding_provider()
    print(f"Embedding provider: {provider}")

    if args.embeddings_only:
        if not (DATA_DIR / "upnote_text.csv").exists():
            print("Error: data/upnote_text.csv not found. Run without --embeddings-only first.")
            sys.exit(1)
        if provider == "google":
            _print_step(3, "Creating Google embeddings and FAISS index (embeddings-only mode)")
            _create_embeddings_google()
        else:
            _print_step(3, "Creating sentence embeddings and FAISS index (embeddings-only mode)")
            print("This may take a long time for large note collections.\n")
            _create_embeddings()
        _print_done(provider)
        return

    # Step 1: .upnx → upnote_text.csv
    _print_step(1, "Creating text dataframe from .upnx backup files")
    _create_dataframe(args.path)

    if provider == "google":
        # Step 2 skipped (no Sudachi tokenization needed)
        # Step 3: Google API embeddings
        _print_step(3, "Creating Google embeddings and FAISS index")
        _create_embeddings_google()
    else:
        # Step 2: upnote_text.csv → upnote_text_split.csv (BM25 tokens)
        _print_step(2, "Tokenizing text for BM25 search")
        _tokenize_for_bm25()

        # Step 3: embeddings.npy + faiss.index
        _print_step(3, "Creating sentence embeddings and FAISS index")
        print("This may take a long time for large note collections.\n")
        _create_embeddings()

    _print_done(provider)


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


def _print_done(provider: str = "local") -> None:
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    out_dir = GOOGLE_DATA_DIR if provider == "google" else LOCAL_DATA_DIR
    print(f"\nGenerated files in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
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
    ids      = df["id"].to_list()
    updates  = df["update"].to_list()
    contents = df["contents"].fill_null("").to_list()
    n_total  = len(ids)

    # Load cache from previous run
    split_path = LOCAL_DATA_DIR / "upnote_text_split.csv"
    cache: dict = {}  # id -> (update, tokens_str)
    if split_path.exists():
        cached_df = pl.read_csv(split_path)
        for cid, cup, ctok in zip(
            cached_df["id"].to_list(),
            cached_df["update"].to_list(),
            cached_df["tokens"].fill_null("").to_list(),
        ):
            cache[cid] = (cup, ctok)
        print(f"Cache loaded: {len(cache)} entries from previous run.")
    else:
        print("No cache found - tokenizing all notes from scratch.")

    # Classify notes
    new_indices = []
    for i, (nid, nup) in enumerate(zip(ids, updates)):
        if nid not in cache or cache[nid][0] != nup:
            new_indices.append(i)

    n_cached = n_total - len(new_indices)
    print(f"  {n_cached} notes reused from cache, {len(new_indices)} notes to tokenize.")

    # Tokenize only new/changed notes
    new_tokens: dict[int, str] = {}
    if new_indices:
        tokenizer = create_tokenizer()
        for i in tqdm(new_indices, desc="Tokenizing", unit="note"):
            new_tokens[i] = tokens_to_string(tokenize_text(contents[i], tokenizer))

    # Assemble in current CSV row order
    tokens_list = []
    for i, (nid, nup) in enumerate(zip(ids, updates)):
        if nid in cache and cache[nid][0] == nup:
            tokens_list.append(cache[nid][1])
        else:
            tokens_list.append(new_tokens[i])

    df.with_columns(pl.Series("tokens", tokens_list)) \
      .select(pl.exclude("contents")) \
      .write_csv(split_path)


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
    meta_path = LOCAL_DATA_DIR / "embedding_meta.csv"
    npy_path  = LOCAL_DATA_DIR / "embeddings.npy"
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
        print("No cache found - encoding all notes from scratch.")

    # Load checkpoint (partial results from a previous interrupted run)
    ckpt_path = LOCAL_DATA_DIR / "embedding_checkpoint.npz"
    ckpt: dict = {}  # id -> (update_ts, np.ndarray embedding)
    if ckpt_path.exists():
        with np.load(ckpt_path, allow_pickle=True) as data:
            ckpt_ids_arr = data["ids"].tolist()
            ckpt_ups_arr = data["updates"].tolist()
            ckpt_embs_arr = data["embeddings"].copy()  # shape (N, dim)
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
        BATCH = 64
        total_batches = (len(new_contents) + BATCH - 1) // BATCH
        print(f"  Loading model (this may take a while)...")
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        print(f"  Model loaded. Encoding {len(new_contents)} notes in {total_batches} batches of up to {BATCH}...")

        # Seed ckpt accumulator with already-loaded checkpoint data
        ckpt_ids_list  = list(ckpt.keys())
        ckpt_ups_list  = [ckpt[k][0] for k in ckpt_ids_list]
        ckpt_embs_list = [ckpt[k][1] for k in ckpt_ids_list]
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
            print(f"  Batch {b + 1}/{total_batches} done - checkpoint saved ({len(ckpt_ids_list)} notes).")

    else:
        if old_embeddings is not None:
            dim = old_embeddings.shape[1]
        print("All notes served from cache/checkpoint - skipping model load.")

    # Assemble in current CSV row order
    print("  Assembling final embedding matrix...")
    final_embeddings = np.empty((n_total, dim), dtype="float32")
    for i, (nid, nup) in enumerate(zip(ids, updates)):
        if nid in cache and cache[nid][0] == nup:
            final_embeddings[i] = old_embeddings[cache[nid][1]]
        else:
            final_embeddings[i] = ckpt[nid][1]

    # Save outputs
    print(f"  Saving embeddings ({n_total} × {dim})...")
    np.save(npy_path, final_embeddings)
    print(f"Saved: {npy_path}")

    print("  Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(final_embeddings)
    out_faiss = LOCAL_DATA_DIR / "faiss.index"
    print("  Saving FAISS index...")
    faiss.write_index(index, str(out_faiss))
    print(f"Saved: {out_faiss}")

    pl.DataFrame({"id": ids, "update": updates}).write_csv(meta_path)
    print(f"Saved: {meta_path}")

    if ckpt_path.exists():
        ckpt_path.unlink()
        print("Checkpoint deleted (run complete).")


def _create_embeddings_google() -> None:
    import faiss
    import numpy as np
    import polars as pl
    from tqdm import tqdm

    from src.config import get_gemini_api_key
    from src.embedding import embed_with_google, GOOGLE_EMBEDDING_DIM

    api_key = get_gemini_api_key()
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env or environment variables.")
        sys.exit(1)

    GOOGLE_DATA_DIR.mkdir(exist_ok=True)

    df = pl.read_csv(DATA_DIR / "upnote_text.csv")
    ids      = df["id"].to_list()
    updates  = df["update"].to_list()
    contents = df["contents"].fill_null("").to_list()
    n_total  = len(ids)

    # Load cache
    meta_path = GOOGLE_DATA_DIR / "embedding_meta.csv"
    npy_path  = GOOGLE_DATA_DIR / "embeddings.npy"
    cache: dict = {}       # id -> (update_ts, old_row_idx)
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
        print("No cache found - encoding all notes from scratch.")

    # Load checkpoint
    ckpt_path = GOOGLE_DATA_DIR / "embedding_checkpoint.npz"
    ckpt: dict = {}  # id -> (update_ts, np.ndarray embedding)
    if ckpt_path.exists():
        with np.load(ckpt_path, allow_pickle=True) as data:
            ckpt_ids_arr  = data["ids"].tolist()
            ckpt_ups_arr  = data["updates"].tolist()
            ckpt_embs_arr = data["embeddings"].copy()
        for cid, cup, cemb in zip(ckpt_ids_arr, ckpt_ups_arr, ckpt_embs_arr):
            ckpt[cid] = (cup, cemb)
        print(f"Checkpoint loaded: {len(ckpt)} partially-encoded notes resumed.")
    else:
        print("No checkpoint found.")

    # Classify notes
    new_indices, new_contents = [], []
    for i, (nid, nup) in enumerate(zip(ids, updates)):
        if nid in cache and cache[nid][0] == nup:
            pass
        elif nid in ckpt and ckpt[nid][0] == nup:
            pass
        else:
            new_indices.append(i)
            new_contents.append(contents[i])

    n_cached = n_total - len(new_indices)
    print(f"  {n_cached} notes reused from cache/checkpoint, {len(new_indices)} notes to encode.")

    # Encode only new notes via Google API
    dim = GOOGLE_EMBEDDING_DIM
    if new_contents:
        ckpt_ids_list  = list(ckpt.keys())
        ckpt_ups_list  = [ckpt[k][0] for k in ckpt_ids_list]
        ckpt_embs_list = [ckpt[k][1] for k in ckpt_ids_list]

        BATCH = 100  # Google API limit
        total_batches = (len(new_contents) + BATCH - 1) // BATCH
        with tqdm(total=len(new_contents), desc="Embedding notes (google)", unit="note") as pbar:
            for b in range(total_batches):
                batch_contents = new_contents[b * BATCH : (b + 1) * BATCH]
                batch_indices  = new_indices [b * BATCH : (b + 1) * BATCH]
                try:
                    batch_emb = embed_with_google(batch_contents, "RETRIEVAL_DOCUMENT", api_key)
                except RuntimeError as e:
                    print(f"\n{e}")
                    print("\nCheckpoint is saved. Re-run to resume from where it stopped.")
                    sys.exit(1)

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
                pbar.update(len(batch_contents))
    else:
        if old_embeddings is not None:
            dim = old_embeddings.shape[1]
        print("All notes served from cache/checkpoint - skipping API calls.")

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
    out_faiss = GOOGLE_DATA_DIR / "faiss.index"
    faiss.write_index(index, str(out_faiss))
    print(f"Saved: {out_faiss}")

    pl.DataFrame({"id": ids, "update": updates}).write_csv(meta_path)
    print(f"Saved: {meta_path}")

    # Minimal upnote_text_split.csv for load_index() compatibility
    split_path = GOOGLE_DATA_DIR / "upnote_text_split.csv"
    pl.DataFrame({"id": ids, "update": updates}).write_csv(split_path)
    print(f"Saved: {split_path}")

    if ckpt_path.exists():
        ckpt_path.unlink()
        print("Checkpoint deleted (run complete).")


if __name__ == "__main__":
    main()
