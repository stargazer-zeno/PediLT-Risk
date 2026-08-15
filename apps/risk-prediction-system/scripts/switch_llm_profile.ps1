param(
    [string]$Profile,
    [switch]$List,
    [switch]$RestartBackend,
    [switch]$NoBackup
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ProfilesDir = Join-Path $ProjectRoot ".env.profiles"
$CommonFile = Join-Path $ProfilesDir "common.env"
$EnvFile = Join-Path $ProjectRoot ".env"

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    foreach ($line in [System.IO.File]::ReadLines($Path)) {
        if ($line -match "^\s*$([regex]::Escape($Name))\s*=\s*(.*)\s*$") {
            return $Matches[1].Trim("'`" ")
        }
    }
    return $null
}

if ($List) {
    if (-not (Test-Path -LiteralPath $ProfilesDir)) {
        Write-Host "No profile directory found: $ProfilesDir"
        exit 0
    }

    Get-ChildItem -LiteralPath $ProfilesDir -Filter "*.env" |
        Where-Object { $_.Name -ne "common.env" } |
        ForEach-Object {
            $profileName = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
            $baseUrl = Get-EnvValue -Path $_.FullName -Name "LLM_BASE_URL"
            $modelName = Get-EnvValue -Path $_.FullName -Name "LLM_MODEL_NAME"
            Write-Host ("{0}`t{1}`t{2}" -f $profileName, $baseUrl, $modelName)
        }
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Profile)) {
    throw "Usage: .\scripts\switch_llm_profile.ps1 -Profile server [-RestartBackend]"
}

if ($Profile -notmatch "^[A-Za-z0-9_.-]+$") {
    throw "Invalid profile name '$Profile'. Use letters, numbers, dot, underscore, or dash only."
}

$ProfileFile = Join-Path $ProfilesDir "$Profile.env"
if (-not (Test-Path -LiteralPath $ProfileFile)) {
    throw "Profile file not found: $ProfileFile. Copy one of the *.env.example files first."
}

if ((Test-Path -LiteralPath $EnvFile) -and -not $NoBackup) {
    $backupFile = "$EnvFile.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
    Copy-Item -LiteralPath $EnvFile -Destination $backupFile -Force
    Write-Host "Backed up previous .env to: $backupFile"
}

$parts = New-Object System.Collections.Generic.List[string]
$parts.Add("# Managed by scripts/switch_llm_profile.ps1")
$parts.Add("# Active LLM profile: $Profile")
$parts.Add("# Sources: .env.profiles/common.env + .env.profiles/$Profile.env")
$parts.Add("")

if (Test-Path -LiteralPath $CommonFile) {
    $parts.Add((Get-Content -LiteralPath $CommonFile -Raw))
    $parts.Add("")
}

$parts.Add((Get-Content -LiteralPath $ProfileFile -Raw))
$content = ($parts -join "`r`n").TrimEnd() + "`r`n"
$encoding = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($EnvFile, $content, $encoding)

$base = Get-EnvValue -Path $EnvFile -Name "LLM_BASE_URL"
$model = Get-EnvValue -Path $EnvFile -Name "LLM_MODEL_NAME"
Write-Host "Active LLM profile: $Profile"
Write-Host "LLM_BASE_URL: $base"
Write-Host "LLM_MODEL_NAME: $model"
Write-Host "LLM_API_KEY: <redacted>"

if ($RestartBackend) {
    Push-Location $ProjectRoot
    try {
        docker compose up -d --force-recreate backend
    }
    finally {
        Pop-Location
    }
}
