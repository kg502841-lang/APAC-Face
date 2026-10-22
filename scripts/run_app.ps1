# Start the FastAPI backend (new window) and the Streamlit dashboard.
#   powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
param([string]$Python = "$PSScriptRoot\..\.venv\Scripts\python.exe", [int]$ApiPort = 8000, [int]$UiPort = 8501)
Set-Location (Resolve-Path "$PSScriptRoot\..")
Start-Process -FilePath $Python -ArgumentList @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "$ApiPort")
$env:APAC_API_URL = "http://127.0.0.1:$ApiPort"
& $Python -m streamlit run frontend/app.py --server.port $UiPort
