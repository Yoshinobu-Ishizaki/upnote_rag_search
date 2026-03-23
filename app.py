import streamlit as st

st.set_page_config(page_title="UpNote RAG Search", layout="wide")

pg = st.navigation([
    st.Page("pages/rag_search.py", title="Search"),
    st.Page("pages/1_Settings.py", title="Settings"),
    st.Page("pages/2_Help.py", title="Help"),
])
pg.run()
