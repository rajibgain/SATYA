@echo off
echo Starting FakeShield Environment Check...
python doctor.py
echo.
echo Starting FakeShield Streamlit App...
streamlit run app.py
pause
