@echo off
call .venv\Scripts\activate.bat && uv run python -m streamlit run app.py
