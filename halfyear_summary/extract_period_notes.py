#!/usr/bin/env python
"""半期評価対象期間の UpNote メモを抽出する。

評価月は 9月・3月。
- 実行月が 4〜9月 → 対象期間はその年の 4月1日 〜 実行日
- 実行月が 10〜12月・1〜3月 → 対象期間は直近の 10月1日 〜 実行日

抽出結果は halfyear_summary/_period_notes.md に書き出す。
これは中間ファイルであり、Claude Code が読んでトピック別に要約するための材料。

Usage:
    uv run python halfyear_summary/extract_period_notes.py
    uv run python halfyear_summary/extract_period_notes.py --today 2026-09-08  # テスト用に基準日を指定
"""
import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = Path(__file__).resolve().parent


def compute_period(today: date) -> tuple[date, date]:
    if 4 <= today.month <= 9:
        start = date(today.year, 4, 1)
    elif today.month >= 10:
        start = date(today.year, 10, 1)
    else:  # 1〜3月
        start = date(today.year - 1, 10, 1)
    return start, today


def main() -> None:
    parser = argparse.ArgumentParser(description="半期分のUpNoteメモを抽出する")
    parser.add_argument("--today", metavar="YYYY-MM-DD", help="基準日（テスト用。省略時は本日）")
    parser.add_argument("--output", metavar="FILE", help="出力先（省略時は halfyear_summary/_period_notes.md）")
    args = parser.parse_args()

    today = date.fromisoformat(args.today) if args.today else date.today()
    start, end = compute_period(today)

    from src.config import get_backup_path
    from src.parser import find_latest_upnx, parse_upnx

    backup_root = get_backup_path()
    upnx_path = find_latest_upnx(backup_root)
    print(f"バックアップ: {upnx_path}", file=sys.stderr)

    df = parse_upnx(upnx_path)

    def _note_date(created) -> date | None:
        s = str(created)[:10] if created else ""
        try:
            return date.fromisoformat(s)
        except ValueError:
            return None

    rows = []
    for r in df.iter_rows(named=True):
        d = _note_date(r["created"])
        if d is not None and start <= d <= end:
            rows.append((d, r))
    rows.sort(key=lambda x: x[0])

    lines = [
        f"# 対象期間: {start.isoformat()} 〜 {end.isoformat()}",
        f"# 抽出ノート数: {len(rows)} 件",
        "",
    ]
    for d, r in rows:
        category = r["category"] or "(カテゴリなし)"
        tags = r["tags"] or ""
        lines.append(f"--- [{d.isoformat()}] {r['fpath']} | category: {category} | tags: {tags} ---")
        lines.append((r["contents"] or "").strip())
        lines.append("")

    output_text = "\n".join(lines)

    out_path = Path(args.output) if args.output else OUT_DIR / "_period_notes.md"
    out_path.write_text(output_text, encoding="utf-8")

    summary_filename = f"{today.year:04d}-{today.month:02d}_summary.md"

    print(f"対象期間: {start.isoformat()} 〜 {end.isoformat()}", file=sys.stderr)
    print(f"抽出ノート数: {len(rows)} 件", file=sys.stderr)
    print(f"抽出結果: {out_path}", file=sys.stderr)
    print(f"要約ファイル名（推奨）: {summary_filename}", file=sys.stderr)


if __name__ == "__main__":
    main()
