"""OneDrive 名刺フォルダ vs UpNote 名刺ノートブック 差分チェック

OneDrive にあって UpNote に未登録の名刺ファイルを一覧出力する。

Usage:
    uv run python check_meishi.py [--output FILE]
"""
import argparse
import gzip
import json
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ONEDRIVE_MEISHI = Path(r"C:\Users\ZT717572\OneDrive - YAMAHA Group\名刺")
NOTEBOOK_NAME = "名刺"


def normalize(s: str) -> str:
    """NFKC 正規化 + 小文字化 + 前後空白除去。比較用。"""
    return unicodedata.normalize("NFKC", s).lower().strip()


def parse_upnx_meishi(upnx_path: Path):
    """upnx を解析して名刺ノートブックに属するノートタイトルと添付ファイル名を返す。

    Returns:
        tuple[set[str], list[tuple[str, str]]]:
            (upnote_stems, debug_titles_and_files)
            upnote_stems: 正規化済みのステム集合
    """
    notebooks: dict[str, dict] = {}
    organizers: list[dict] = []
    nb_lists: dict[str, dict] = {}
    notes: dict[str, dict] = {}
    files: dict[str, str] = {}  # id -> name

    with gzip.open(upnx_path, "rb") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line == b"version:2":
                continue
            obj = json.loads(line)
            t, d = obj["type"], obj["data"]
            if t == "notebooks":
                notebooks[d["id"]] = {"title": d.get("title", ""), "parent": d.get("parent")}
            elif t == "organizers" and not d.get("deleted"):
                if d.get("noteId") and d.get("notebookId"):
                    organizers.append(d)
            elif t == "lists":
                if d["id"].startswith("notebooks_"):
                    nb_lists[d["id"]] = d
            elif t == "notes":
                if not d.get("trashed") and not d.get("deleted"):
                    file_ids = d.get("fileIds", {}).get("__value__", []) if isinstance(d.get("fileIds"), dict) else []
                    notes[d["id"]] = {"title": d.get("title") or "", "fileIds": file_ids}
            elif t == "files":
                name = d.get("name") or ""
                if name:
                    files[d["id"]] = name

    # 名刺ノートブック ID を title で検索
    meishi_nb_ids: set[str] = set()
    for nb_id, nb in notebooks.items():
        if normalize(nb["title"]) == normalize(NOTEBOOK_NAME):
            meishi_nb_ids.add(nb_id)

    if not meishi_nb_ids:
        print(f"[警告] UpNote に「{NOTEBOOK_NAME}」ノートブックが見つかりません。", file=sys.stderr)
        return set(), []

    # organizers から名刺ノートブックに属するノート ID を収集
    meishi_note_ids: set[str] = set()
    for org in organizers:
        if org["notebookId"] in meishi_nb_ids:
            meishi_note_ids.add(org["noteId"])

    # nb_lists からも収集
    for lid, le in nb_lists.items():
        nb_uuid = lid[len("notebooks_"):]
        if nb_uuid not in meishi_nb_ids:
            continue
        content = le.get("content", "")
        try:
            note_ids = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(note_ids, list):
            meishi_note_ids.update(note_ids)

    # 各ノートのタイトルと添付ファイル名のステムを収集
    upnote_stems: set[str] = set()
    debug_entries: list[tuple[str, str]] = []  # (kind, display_name)

    for note_id in meishi_note_ids:
        note = notes.get(note_id)
        if not note:
            continue
        title = note["title"]
        if title:
            stem = Path(title).stem if "." in title else title
            upnote_stems.add(normalize(stem))
            debug_entries.append(("title", title))
        for fid in note["fileIds"]:
            fname = files.get(fid, "")
            if fname:
                fstem = Path(fname).stem
                upnote_stems.add(normalize(fstem))
                debug_entries.append(("file", fname))

    return upnote_stems, debug_entries


def main():
    parser = argparse.ArgumentParser(description="OneDrive名刺 vs UpNote名刺 差分チェック")
    parser.add_argument("--output", metavar="FILE", help="結果をファイルにも保存する")
    parser.add_argument("--debug", action="store_true", help="UpNote側の登録内容も表示する")
    args = parser.parse_args()

    # バックアップパスを config から取得
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.parser import find_latest_upnx
    from src.config import get_backup_path

    backup_root = get_backup_path()
    upnx_path = find_latest_upnx(backup_root)
    print(f"バックアップ: {upnx_path}", file=sys.stderr)

    upnote_stems, debug_entries = parse_upnx_meishi(upnx_path)

    if args.debug:
        print(f"\n--- UpNote 名刺ノートブック内の登録 ({len(debug_entries)} 件) ---")
        for kind, name in sorted(debug_entries, key=lambda x: x[1]):
            print(f"  [{kind}] {name}")

    # OneDrive PDF 一覧
    if not ONEDRIVE_MEISHI.exists():
        print(f"[エラー] OneDrive フォルダが見つかりません: {ONEDRIVE_MEISHI}", file=sys.stderr)
        sys.exit(1)

    onedrive_pdfs = sorted(ONEDRIVE_MEISHI.glob("*.pdf"), key=lambda p: p.name)

    # 差分計算
    missing: list[Path] = []
    for pdf in onedrive_pdfs:
        stem = pdf.stem
        if normalize(stem) not in upnote_stems:
            missing.append(pdf)

    # 出力
    lines: list[str] = []
    lines.append(f"OneDrive にあって UpNote に未登録の名刺ファイル: {len(missing)} 件 / 全 {len(onedrive_pdfs)} 件\n")
    lines.append("| # | ファイル名 |")
    lines.append("|---|-----------|")
    for i, pdf in enumerate(missing, 1):
        lines.append(f"| {i} | {pdf.name} |")

    output_text = "\n".join(lines)
    print(output_text)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"\n結果を保存しました: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
