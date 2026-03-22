# 全てのデータを読込み、文を分かち書きにして別ファイルに出力する。

import polars as pl 
import sudachipy 
import os, os.path
import unicodedata
import re

poc = ["名詞","形容詞","動詞","接尾辞","接頭辞"]

def strtrans(s):
    """キーワードを以下の手順で正規化されたスペース区切りの文に変換する。
        - ユニコード文字列として正規化(全角英数は半角へ、半角カナなどは全角へ)
        - 英数字は小文字化
        - 図番に相当するas-j-99999の形のコードをasj99999型に変換
        - sudachiで分かち書き
        - 名詞・形容詞・動詞・"接尾辞"のみを抽出
        - 正規化トークンに変換
        一連の変換を関数にして、それを各行のデータに適用する。
    """
    tokens = [] # normalized token
    s1 = unicodedata.normalize("NFKC",s).lower()
    s2 = re.sub(r"([a-z]{2})-([djm])-([0-9]{4,5})", r"\1\2\3",s1) # replace drawing number
    tkn = tokenizer.tokenize(s2)
    for t in tkn:
        w = t.surface()
        if not w.isspace():
            if t.part_of_speech()[0] in poc:
                n = t.normalized_form()
                tokens.append(n)
    
    return (tokens)

def strtrans_text(s):
    tokens = []
    if len(s) > 2**14 -1:
        ss = s.split("\n")
        for sl in ss:
            tk =  strtrans(sl) 
            tokens = tokens + tk
    else:
        tokens = strtrans(s)

    return (" ".join(tokens))

if __name__ == "__main__":

    # change working directory
    if os.path.basename(os.getcwd()) == "script":
        os.chdir("..")

    # read original dataframe
    dfm = pl.read_csv("data/upnote_text.csv")

    # dict = sudachipy.Dictionary(config_path="../sudachi/sudachi.json")
    dict = sudachipy.Dictionary()
    tokenizer = dict.create()
    tokens_list = [strtrans_text(s) for s in dfm["contents"].to_list()]
    dfm2 = dfm.with_columns(pl.Series("tokens", tokens_list))

    dfm2.select(pl.exclude("contents")).write_csv("data/upnote_text_split.csv")

