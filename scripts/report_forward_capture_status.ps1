param(
    [string]$DbPath = "",
    [string]$CollectSummary = "",
    [switch]$Json
)

. "$PSScriptRoot\_predicciones.ps1"

$repoRoot = Get-PrediccionesRepoRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$effectiveDbPath = if ([string]::IsNullOrWhiteSpace($DbPath)) {
    Join-Path $repoRoot "data\polymarket_shadow.sqlite"
} else {
    $DbPath
}

$effectiveCollectSummary = $CollectSummary
if ([string]::IsNullOrWhiteSpace($effectiveCollectSummary)) {
    $pointer = Join-Path $repoRoot "outputs\latest_polymarket_collect.txt"
    if (Test-Path -LiteralPath $pointer) {
        $effectiveCollectSummary = (Get-Content -LiteralPath $pointer -Raw).Trim()
    }
}

$env:PYTHONPATH = Join-Path $repoRoot "src"
$argsList = @(
    "-m", "predicciones.offline_decision_region_review",
    "monitor",
    "--db-path", $effectiveDbPath
)
if (-not [string]::IsNullOrWhiteSpace($effectiveCollectSummary)) {
    $argsList += @("--collect-summary", $effectiveCollectSummary)
}
if ($Json) {
    $argsList += "--json"
}

& $python @argsList
exit $LASTEXITCODE
