import argparse
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_backup_root(cli_path):
    if cli_path:
        return Path(cli_path)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import get_backup_path
    return get_backup_path()


def find_latest_upnx(backup_root):
    files = sorted((backup_root / "data").glob("*.upnx"))
    if not files:
        raise FileNotFoundError(f"No .upnx in {backup_root}/data")
    return files[-1]


def parse_upnx(upnx_path):
    notebooks, organizers, notes = {}, [], []
    with gzip.open(upnx_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line or line == b"version:2":
                continue
            obj = json.loads(line)
            t, d = obj["type"], obj["data"]
            if t == "notebooks":
                notebooks[d["id"]] = d.get("title", "")
            elif t == "organizers" and not d.get("deleted"):
                if d.get("noteId") and d.get("notebookId"):
                    organizers.append(d)
            elif t == "notes":
                notes.append(d)

    note_to_nb = {}
    for org in organizers:
        nid = org["noteId"]
        if nid not in note_to_nb:
            note_to_nb[nid] = notebooks.get(org["notebookId"], "")

    rows = []
    for d in notes:
        if d.get("trashed") or d.get("deleted"):
            continue
        doc_id = d["id"]
        title = d.get("title") or ""
        text = d.get("text") or ""
        update_dt = datetime.fromtimestamp(d["updatedAt"] / 1000) if d.get("updatedAt") else ""
        create_dt = datetime.fromtimestamp(d["createdAt"] / 1000) if d.get("createdAt") else ""
        category = note_to_nb.get(doc_id, "")
        tags = "|".join(d.get("tagLinks", {}).get("__value__", []))
        contents = (title + "\n" + text).strip()
        rows.append([doc_id, title, update_dt, create_dt, category, tags, contents])

    return pl.DataFrame(rows, schema=["id", "fpath", "update", "created", "category", "tags", "contents"], orient="row")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create text dataframe from UpNote .upnx backup")
    parser.add_argument("--path", type=str, help="Path to UpNote backup root (folder containing data/)")
    args = parser.parse_args()

    backup_root = get_backup_root(args.path)
    upnx = find_latest_upnx(backup_root)
    print(f"Reading: {upnx.name}")
    df = parse_upnx(upnx)
    print(f"Notes: {len(df)} (active)")
    out = PROJECT_ROOT / "data" / "upnote_text.csv"
    out.parent.mkdir(exist_ok=True)
    df.write_csv(out)
    print(f"Saved: {out}")
