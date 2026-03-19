"""BM25 keyword search page (migrated from keyword_search/keyword_search.py)."""
import sys
from pathlib import Path

import polars as pl
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

# Ensure project root is in path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_search import build_bm25, load_split_data
from src.tokenizer import POC, create_tokenizer, tokenize_text, tokenize_with_surface

DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="Keyword Search", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def _tokenizer():
    return create_tokenizer()


@st.cache_data
def _load_split() -> pl.DataFrame:
    return load_split_data(DATA_DIR).with_columns(pl.lit(0.0).alias("score"))


@st.cache_data
def _load_text() -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / "upnote_text.csv").select(["id", "contents"])


@st.cache_resource
def _bm25(_df: pl.DataFrame):
    return build_bm25(_df["tokens"])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def strtrans(s: str) -> list[str]:
    tokenizer = _tokenizer()
    return tokenize_text(s, tokenizer)


def get_original_text(dfm_org: pl.DataFrame, doc_id: str, kwd: list[str]) -> str:
    row = dfm_org.filter(pl.col("id") == doc_id)
    if row.is_empty():
        return ""
    txt = row["contents"][0] or ""

    if not kwd:
        return txt

    tokenizer = _tokenizer()
    tokens = tokenize_with_surface(txt, tokenizer)
    tlist = []
    for t in tokens:
        w = t.surface()
        if w.isspace():
            tlist.append("\n\n")
        else:
            if t.part_of_speech()[0] in POC and t.normalized_form() in kwd:
                tlist.append(f":red[{t.normalized_form()}]")
            else:
                tlist.append(w)
    return "".join(tlist)


def filter_dataframe(df: pl.DataFrame, bm25, kwd: list[str]) -> pl.DataFrame:
    if not kwd:
        return df

    scores = bm25.get_scores(kwd)
    df2 = df.with_columns(pl.Series("score", scores))
    return df2.filter(pl.col("score") > 0).sort("score", descending=True)


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------

try:
    dfm = _load_split()
    dfm_org = _load_text()
    bm25 = _bm25(dfm)
    data_ok = True
except FileNotFoundError:
    data_ok = False

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "keyword" not in st.session_state:
    st.session_state["keyword"] = []


def update_page():
    st.session_state["keyword"] = strtrans(st.session_state.keywordinput)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Side Panel")
    st.text_input("Search text for keywords", key="keywordinput", on_change=update_page)
    st.text_input(
        "Normalized keywords",
        disabled=True,
        value=" ".join(st.session_state.keyword),
    )
    if data_ok:
        showcols = st.multiselect(
            "Columns",
            dfm.columns,
            default=["fpath", "created", "category", "tags", "tokens", "score"],
        )

st.header("Upnote Keyword Search")

if not data_ok:
    st.error(
        "Data files not found. Please run `python preprocess.py` first, "
        "then restart the app."
    )
    st.stop()

gb = GridOptionsBuilder.from_dataframe(dfm.head().to_pandas())
gb.configure_selection(selection_mode="single", use_checkbox=False)
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
gb.configure_default_column(initialHide=True)
gb.configure_columns(showcols, hide=False)
gb.configure_column("tokens", width=500)
gb.configure_columns(["tags", "score"], width=100)
gridOptions = gb.build()

df_filtered = filter_dataframe(dfm, bm25, st.session_state.keyword)

grid_response = AgGrid(
    df_filtered.to_pandas(),
    gridOptions=gridOptions,
    enable_enterprise_modules=True,
    allow_unsafe_jscode=True,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
)

selected_rows = grid_response["selected_rows"]

st.subheader("Normalized text of selected row")
with st.container(height=250):
    if selected_rows is not None and len(selected_rows) > 0:
        iid = selected_rows[0]["id"]
        txt = get_original_text(dfm_org, iid, st.session_state.keyword)
        st.markdown(txt)
    else:
        st.markdown("")
