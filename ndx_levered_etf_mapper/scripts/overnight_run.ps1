param(
  [switch]$Commit
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repo "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "overnight_$stamp.log"

function Log($s) {
  Add-Content -Path $log -Value $s
}

function Run($cmd) {
  Log "`n$ $cmd"
  try {
    $out = Invoke-Expression $cmd 2>&1 | Out-String
    Log $out
  } catch {
    Log $_.Exception.Message
  }
}

Run "python scripts/mh_doctor.py"
Run "python -m py_compile app/streamlit_app.py scripts/decisions_listener.py"
Run "python scripts/todo_status_gen.py"
Run "python scripts/todo_mega_sprint_gen.py"
Run "python scripts/todo_mega_status_gen.py"
Run "pytest -q"
Run "python scripts/make_debug_bundle.py"
Run "git status --porcelain"

if ($Commit) {
  # stage safe artifacts (never secrets)
  Run "git add -- TODO_STATUS.md TODO_STATUS_SCAFFOLDS.md TODO_STATUS_QA.md TODO_MEGA_SPRINT.md TODO_MEGA_SPRINT_STATUS.md"
  Run "git add -- $log"
  Run "git commit -m \"Overnight runner artifacts ($stamp)\""
}

Write-Host $log
