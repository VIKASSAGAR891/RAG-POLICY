#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "Setup complete. Edit .env, add PDFs under data/raw/, then run:"
echo "  python scripts/ingest.py"
echo "  streamlit run app.py"
