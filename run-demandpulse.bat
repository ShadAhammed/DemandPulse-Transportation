@echo off
cd /d "%~dp0"
echo Starting DemandPulse from: %CD%
echo Open: http://localhost:8502
echo Expected tabs: Data ^& Architecture ^| Model ^| Overview
echo ExplainAI / AIExplanator usually runs on http://localhost:8501
streamlit run app.py --server.port 8502
