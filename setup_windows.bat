@echo off
echo Creating Python virtual environment...
py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Setup complete.
echo 1. Edit .env and paste your GROQ_API_KEY.
echo 2. Put PDFs under data\raw\
echo 3. Run: python scripts\ingest.py
echo 4. Run: streamlit run app.py
pause
