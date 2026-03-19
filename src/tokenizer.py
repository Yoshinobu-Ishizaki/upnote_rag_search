"""Shared Sudachi tokenizer for BM25 and preprocessing."""
import re
import unicodedata

import sudachipy

POC = ["名詞", "形容詞", "動詞", "接尾辞", "接頭辞"]


def create_tokenizer():
    """Create and return a Sudachi tokenizer instance."""
    dictionary = sudachipy.Dictionary()
    return dictionary.create()


def tokenize_text(text: str, tokenizer) -> list[str]:
    """Tokenize text into normalized tokens, handling Sudachi's 2^14 char limit.

    Returns a list of normalized word strings (nouns, adjectives, verbs, etc.).
    """
    if len(text) > 2**14 - 1:
        tokens: list[str] = []
        for line in text.split("\n"):
            tokens.extend(_tokenize_chunk(line, tokenizer))
        return tokens
    return _tokenize_chunk(text, tokenizer)


def _tokenize_chunk(s: str, tokenizer) -> list[str]:
    s1 = unicodedata.normalize("NFKC", s).lower()
    s2 = re.sub(r"([a-z]{2})-([djm])-([0-9]{4,5})", r"\1\2\3", s1)
    tkn = tokenizer.tokenize(s2)
    tokens = []
    for t in tkn:
        w = t.surface()
        if not w.isspace():
            if t.part_of_speech()[0] in POC:
                tokens.append(t.normalized_form())
    return tokens


def tokenize_with_surface(text: str, tokenizer) -> list:
    """Return raw token objects for display (used in keyword highlighting)."""
    if len(text) > 2**14 - 1:
        all_tokens = []
        for line in text.split("\n"):
            s1 = unicodedata.normalize("NFKC", line).lower()
            s2 = re.sub(r"([a-z]{2})-([djm])-([0-9]{4,5})", r"\1\2\3", s1)
            all_tokens.extend(tokenizer.tokenize(s2))
        return all_tokens
    s1 = unicodedata.normalize("NFKC", text).lower()
    s2 = re.sub(r"([a-z]{2})-([djm])-([0-9]{4,5})", r"\1\2\3", s1)
    return tokenizer.tokenize(s2)


def tokens_to_string(tokens: list[str]) -> str:
    return " ".join(tokens)
