# Run the FastAPI backend locally on Windows.
# Usage:  ./scripts/run_backend.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
$envFile = Join-Path $root ".env"

if (Test-Path $envFile) {
    Write-Host "Loading .env..." -ForegroundColor Cyan
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $idx = $line.IndexOf("=")
            $key = $line.Substring(0, $idx).Trim()
            $value = $line.Substring($idx + 1).Trim()
            if (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            ) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtualenv..." -ForegroundColor Cyan
    python -m venv (Join-Path $root ".venv")
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r (Join-Path $root "backend\requirements.txt")
}

$schema = Join-Path $root "artifacts\datasets\schema.json"
if (-not (Test-Path $schema)) {
    throw "Bundled ML schema is missing: $schema"
}

$env:PYTHONUTF8 = "1"
Push-Location (Join-Path $root "backend")
& $venvPy -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
Pop-Location
