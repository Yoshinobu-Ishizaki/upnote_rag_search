"""Extract date range from a Japanese question using Claude."""
import datetime
import json
import logging

import anthropic

from src.config import get_claude_model

logger = logging.getLogger(__name__)


def extract_date_range(question: str, api_key: str, max_note_date: datetime.date | None = None) -> dict | None:
    """Return {"date_from": date, "date_to": date} or None if no date range found."""
    today = datetime.date.today()
    max_note_info = (
        f"ノートの最新作成日は {max_note_date.isoformat()} です。"
        if max_note_date
        else ""
    )
    system_prompt = (
        f"今日の日付は {today.isoformat()} です。{max_note_info}"
        "ユーザーの質問に含まれる日付・期間の表現を抽出し、絶対的な日付範囲に変換してください。"
        "「最新日付」「最新の日付」などはノートの最新作成日を指します。"
        "必ず以下のJSONのみを返してください（説明文不要）:\n"
        '{"has_date_range": true/false, "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}\n'
        "日付表現がない場合は has_date_range: false を返してください。"
        "date_from と date_to は has_date_range が true の場合のみ含めてください。"
        "範囲は常に包含（inclusive）にしてください。"
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=get_claude_model(),
            max_tokens=64,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        answer_text = next(
            (block.text for block in response.content if block.type == "text"),
            "",
        )
        result = json.loads(answer_text)
        if not result.get("has_date_range"):
            return None
        date_from = datetime.date.fromisoformat(result["date_from"])
        date_to = datetime.date.fromisoformat(result["date_to"])
        if date_from > date_to:
            date_from, date_to = date_to, date_from
        return {"date_from": date_from, "date_to": date_to}
    except Exception:
        logger.warning("Date extraction failed, using manual filter", exc_info=True)
        return None
