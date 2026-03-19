---
name: upnx_reader_plan
description: Plan to replace markdown-file parsing with UpNote native .upnx binary format reader in 01_df-creation.py
type: project
---

**Next task: rewrite `script/01_df-creation.py` to read `.upnx` (UpNote native backup) instead of `.md` files.**

Why: The UpNote backup at `D:\UpNote Backup\zRjPQTYkYdVy58nALydvZJntqA72\` stores notes as `.upnx` files. The markdown files in that backup have no YAML frontmatter, so the old `01_df-creation.py` would produce empty `contents` for all notes.

## .upnx format

- Files live in `{backup_root}/data/*.upnx`
- Each file = gzip-compressed NDJSON; first line is `version:2`
- 17 files total, each is a **full backup snapshot** (not incremental) — use the one with the largest timestamp prefix filename
- Record types: `notes`, `organizers`, `notebooks`, `files`, `tags`, `lists`, `filters`, `user`
- Active notes in latest file: 28,305 (50 trashed, 0 deleted)

## Note record fields

```json
{
  "type": "notes",
  "data": {
    "id": "uuid",
    "title": "note title (UTF-8)",
    "text": "plain text body (UTF-8, no HTML/markdown)",
    "html": "HTML version (not needed)",
    "updatedAt": 1386033538000,  // milliseconds since epoch
    "createdAt": 1343109539000,
    "trashed": false,
    "deleted": false,
    "tagLinks": {"__type__": "Set", "__value__": ["tag1", "tag2"]},
    "notebookLinks": {"__type__": "Set", "__value__": ["notebook-id"]}
  }
}
```

## Notebook linkage

`organizers` records link notes to notebooks:
```json
{"type": "organizers", "data": {"noteId": "...", "notebookId": "..."}}
```
`notebooks` records have `id` → `title`.

## Implementation plan for `script/01_df-creation.py`

```python
import gzip, json
from datetime import datetime
from pathlib import Path
import polars as pl
import sys, argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_backup_root(cli_path):
    if cli_path: return Path(cli_path)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import get_backup_path
    return get_backup_path()

def find_latest_upnx(backup_root):
    files = sorted((backup_root / "data").glob("*.upnx"))
    if not files: raise FileNotFoundError(f"No .upnx in {backup_root}/data")
    return files[-1]

def parse_upnx(upnx_path):
    notebooks, organizers, notes = {}, [], []
    with gzip.open(upnx_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line or line == b"version:2": continue
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
        if d.get("trashed") or d.get("deleted"): continue
        doc_id = d["id"]
        title = d.get("title") or ""
        text = d.get("text") or ""
        update_dt = datetime.fromtimestamp(d["updatedAt"] / 1000) if d.get("updatedAt") else ""
        create_dt = datetime.fromtimestamp(d["createdAt"] / 1000) if d.get("createdAt") else ""
        category = note_to_nb.get(doc_id, "")
        tags = "|".join(d.get("tagLinks", {}).get("__value__", []))
        contents = (title + "\n" + text).strip()
        rows.append([doc_id, title, update_dt, create_dt, category, tags, contents])

    return pl.DataFrame(rows, schema=["id","fpath","update","created","category","tags","contents"], orient="row")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
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
```

## config.ini path

`UPNOTE_BACKUP_PATH` should point to backup ROOT (the folder containing `data/`):
```
UPNOTE_BACKUP_PATH=D:\UpNote Backup\zRjPQTYkYdVy58nALydvZJntqA72
```

NOT the `Markdown/General Space` subfolder.

## What's already done

The full overhaul is COMPLETE (on branch `feat-llmqa`):
- `app.py`, `preprocess.py`, `pyproject.toml`, `config.ini`, `.env`
- `src/config.py`, `src/tokenizer.py`, `src/bm25_search.py`, `src/embedding.py`, `src/hybrid_search.py`
- `pages/1_Keyword_Search.py`, `pages/2_RAG_Search.py`, `pages/3_Settings.py`
- `app.bat`, `app.sh`
- uv venv recreated (Python 3.10.18, 86 packages)
- Only `script/01_df-creation.py` needs the upnx rewrite

## After implementing

Run: `python preprocess.py` (full run, all 3 steps)
Expected: `data/upnote_text.csv` with 28,305 rows, non-empty `contents`, readable `fpath` (note titles)
