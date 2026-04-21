param(
    [string]$RetroRunDir = "",
    [string]$PolicyBundle = ""
)

. "$PSScriptRoot\_predicciones.ps1"

$repoRoot = Get-PrediccionesRepoRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$effectiveRetroRunDir = $RetroRunDir
if ([string]::IsNullOrWhiteSpace($effectiveRetroRunDir)) {
    $pointer = Join-Path $repoRoot "outputs\latest_polymarket_retro.txt"
    if (-not (Test-Path -LiteralPath $pointer)) {
        throw "No existe outputs\latest_polymarket_retro.txt. Ejecuta primero backtest-polymarket-retro."
    }
    $effectiveRetroRunDir = (Get-Content -LiteralPath $pointer -Raw).Trim()
}

$effectivePolicyBundle = $PolicyBundle
if ([string]::IsNullOrWhiteSpace($effectivePolicyBundle)) {
    $candidate = Join-Path $effectiveRetroRunDir "policy_bundle.json"
    if (Test-Path -LiteralPath $candidate) {
        $effectivePolicyBundle = $candidate
    }
}

$env:PYTHONPATH = Join-Path $repoRoot "src"
$argsList = @(
    "-m", "predicciones.offline_decision_region_review",
    "review",
    "--retro-run-dir", $effectiveRetroRunDir,
    "--outputs-dir", (Join-Path $repoRoot "outputs")
)
if (-not [string]::IsNullOrWhiteSpace($effectivePolicyBundle)) {
    $argsList += @("--policy-bundle", $effectivePolicyBundle)
}

& $python @argsList
exit $LASTEXITCODE
