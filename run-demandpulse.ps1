# Launch DemandPulse on port 8502 (ExplainAI often uses 8501)
Set-Location $PSScriptRoot
Write-Host "Starting DemandPulse from: $PWD"
Write-Host "Open: http://localhost:8502"
Write-Host "Expected tabs: Data & Setup | Model & Test Results | Executive Briefing"
streamlit run app.py --server.port 8502 --server.headless true
