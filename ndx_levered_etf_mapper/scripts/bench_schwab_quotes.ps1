param(
  [string]$Symbols = "SPY,QQQ,TSLA,AAPL,NVDA",
  [int]$Iters = 20,
  [int]$SleepMs = 3000
)

$syms = $Symbols.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ }
$tokenPath = Resolve-Path "data\schwab_tokens.json" -ErrorAction Stop
$tok = Get-Content $tokenPath -Raw | ConvertFrom-Json
$access = $tok.access_token
if (-not $access) { throw "No access_token in $tokenPath" }

function Percentile([double[]]$xs, [double]$p) {
  if ($xs.Length -eq 0) { return [double]::NaN }
  $ys = $xs | Sort-Object
  $k = ($ys.Length - 1) * $p
  $f = [int][Math]::Floor($k)
  $c = [int][Math]::Min($f + 1, $ys.Length - 1)
  if ($f -eq $c) { return $ys[$f] }
  return $ys[$f] * ($c - $k) + $ys[$c] * ($k - $f)
}

$lats = New-Object System.Collections.Generic.List[double]
$fail = 0

for ($i=0; $i -lt $Iters; $i++) {
  $t0 = [System.Diagnostics.Stopwatch]::StartNew()
  $ok = $true
  try {
    $url = "https://api.schwabapi.com/marketdata/v1/quotes?symbols=$([uri]::EscapeDataString(($syms -join ',')))"
    $resp = Invoke-RestMethod -Uri $url -Headers @{ Authorization = "Bearer $access"; Accept = "application/json" } -Method GET -TimeoutSec 30
  } catch {
    $ok = $false
  }
  $t0.Stop()
  if ($ok) { $lats.Add([double]$t0.Elapsed.TotalMilliseconds) } else { $fail += 1 }
  Start-Sleep -Milliseconds $SleepMs
}

$xs = $lats.ToArray()
$out = [ordered]@{
  lang = "powershell"
  iters = $Iters
  ok = $xs.Length
  fail = $fail
  symbols = $syms
  lat_ms = [ordered]@{
    min = if ($xs.Length) { ($xs | Measure-Object -Minimum).Minimum } else { $null }
    mean = if ($xs.Length) { ($xs | Measure-Object -Average).Average } else { $null }
    p50 = if ($xs.Length) { (Percentile $xs 0.5) } else { $null }
    p95 = if ($xs.Length) { (Percentile $xs 0.95) } else { $null }
    max = if ($xs.Length) { ($xs | Measure-Object -Maximum).Maximum } else { $null }
  }
}

$out | ConvertTo-Json -Depth 6
