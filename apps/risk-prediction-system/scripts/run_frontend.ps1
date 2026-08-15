# Run the React + Vite dev server (proxies /api to http://127.0.0.1:8000).
# Usage:  ./scripts/run_frontend.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "frontend")
if (-not (Test-Path "node_modules")) {
    npm install --no-audit --no-fund
}
npm run dev
Pop-Location
