import argparse
import glob
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_backup_path(cli_path: str | None) -> Path:
    """Resolve backup path from CLI arg or config.ini."""
    if cli_path:
        return Path(cli_path)
    # Fallback to config.ini
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.config import get_backup_path as _cfg_path

        return _cfg_path()
    except Exception:
        return PROJECT_ROOT / "UpNote" / "General Space"


def convtxt(path: Path):
    update_dt = ""
    create_dt = ""
    cat_mode = False
    isheader = True
    categories = []
    bdytxt = []
    tags = []

    with open(path, "r", encoding="utf-8") as f:
        txt = f.readlines()

    for i, line in enumerate(txt):
        if i > 0:
            s = line.rstrip()
            if s.startswith("date:"):
                s2 = s.replace("date: ", "")
                update_dt = datetime.strptime(s2, "%Y-%m-%d %H:%M:%S")
            elif s.startswith("created: "):
                s2 = s.replace("created: ", "")
                create_dt = datetime.strptime(s2, "%Y-%m-%d %H:%M:%S")
            elif s.startswith("categories:"):
                cat_mode = True
            elif s == "---":
                isheader = False
                cat_mode = False
                continue
            else:
                if cat_mode:
                    s2 = re.sub(r"^- ", "", s)
                    categories.append(s2)

            if not isheader:
                if bool(re.match(r"#+ ", s)):
                    s2 = re.sub(r"#+ ", "", s)
                    bdytxt.append(s2)
                elif bool(re.match(r"^[\s\*]+$", s)):
                    bdytxt.append("")
                elif bool(re.findall(r"#\w+", s)):
                    for m in re.findall(r"#\w+", s):
                        s2 = re.sub(r"^#", "", m)
                        if not s2.isdigit():
                            tags.append(s2)
                else:
                    bdytxt.append(s)

    cat_str = "|".join(categories)
    contents = "\n".join(bdytxt)
    tags_str = "|".join(tags)

    fpath = path.name

    sha1 = hashlib.sha1()
    sha1.update(fpath.encode("utf-8"))
    doc_id = sha1.hexdigest()

    yield [doc_id, fpath, update_dt, create_dt, cat_str, tags_str, contents]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create text dataframe from UpNote markdown files")
    parser.add_argument("--path", type=str, help="Path to UpNote backup folder")
    args = parser.parse_args()

    upnote_path = get_backup_path(args.path)
    print(f"Backup path: {upnote_path}")

    if not upnote_path.exists():
        print(f"Error: path does not exist: {upnote_path}")
        sys.exit(1)

    files = list(upnote_path.glob("*.md"))
    maxi = len(files)
    print(f"Found {maxi} markdown files")

    alldata = []
    for i, f in enumerate(files):
        for output in convtxt(f):
            alldata.append(output)

        if (i % 100) == 0:
            print(f"\x1b[2K\r{i}/{maxi}: {f.name}", end="\r")

    print(f"\x1b[2K\rProcessed {maxi} files")

    dfm = pl.DataFrame(
        alldata,
        schema=["id", "fpath", "update", "created", "category", "tags", "contents"],
        orient="row",
    )

    out_path = PROJECT_ROOT / "data" / "upnote_text.csv"
    out_path.parent.mkdir(exist_ok=True)
    dfm.write_csv(out_path)
    print(f"Saved: {out_path} ({len(dfm)} rows)")
