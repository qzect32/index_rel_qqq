param(
  [string]$Symbols = "SPY,QQQ,TSLA,AAPL,NVDA",
  [int]$Iters = 20,
  [int]$SleepMs = 3000
)

Write-Host "[1/4] Refresh token (Python)" -ForegroundColor Green
python scripts/bench_schwab_refresh.py

Write-Host "[2/4] Python quotes bench" -ForegroundColor Green
python scripts/bench_schwab_quotes.py --symbols $Symbols --iters $Iters --sleep-ms $SleepMs

Write-Host "[3/4] Node quotes bench" -ForegroundColor Green
node scripts/bench_schwab_quotes.js --symbols $Symbols --iters $Iters --sleep-ms $SleepMs

Write-Host "[4/4] PowerShell quotes bench" -ForegroundColor Green
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bench_schwab_quotes.ps1 -Symbols $Symbols -Iters $Iters -SleepMs $SleepMs
