param(
    [string]$LaneId = "football_1x2_global",
    [switch]$IncludeGoals,
    [int]$MaxEvents = 5,
    [int]$MaxMarkets = 0,
    [string]$CapturePriority = "missing_or_stale",
    [int]$FreshnessSeconds = 3600,
    [int]$MaxStaleAgeSeconds = 0,
    [string]$ModelPath = "",
    [switch]$IncludePolicyReadyOnly,
    [switch]$DryRun
)

. (Join-Path $PSScriptRoot "_predicciones.ps1")

function Invoke-LaneCycleCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command,

        [string[]]$Arguments = @()
    )

    $commandParts = @($Command) + @($Arguments)
    $rendered = "predicciones $($commandParts -join ' ')".Trim()
    if ($DryRun) {
        Write-Host "[dry-run] $rendered"
        return
    }

    Write-Host "[run] $rendered"
    $repoRoot = Get-PrediccionesRepoRoot
    $venvExe = Join-Path $repoRoot ".venv\Scripts\predicciones.exe"
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvExe) {
        & $venvExe @commandParts
    } elseif (Test-Path -LiteralPath $venvPython) {
        $srcPath = Join-Path $repoRoot "src"
        if ($env:PYTHONPATH) {
            $env:PYTHONPATH = "$srcPath$([IO.Path]::PathSeparator)$env:PYTHONPATH"
        } else {
            $env:PYTHONPATH = $srcPath
        }
        & $venvPython -m predicciones.cli @commandParts
    } else {
        & predicciones @commandParts
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $rendered"
    }
}

function New-CaptureArguments {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetLane,

        [bool]$PolicyReadyOnly
    )

    $arguments = @(
        "--lane-id", $TargetLane,
        "--capture-priority", $CapturePriority,
        "--freshness-seconds", [string]$FreshnessSeconds
    )
    if ($MaxEvents -gt 0) {
        $arguments += @("--max-events", [string]$MaxEvents)
    }
    if ($MaxMarkets -gt 0) {
        $arguments += @("--max-markets", [string]$MaxMarkets)
    }
    if ($MaxStaleAgeSeconds -gt 0) {
        $arguments += @("--max-stale-age-seconds", [string]$MaxStaleAgeSeconds)
    }
    if ($PolicyReadyOnly) {
        $arguments += "--include-policy-ready-only"
    }
    return $arguments
}

function Invoke-MarketLaneCycle {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetLane,

        [bool]$PolicyReadyOnly
    )

    Write-Host "Multi-market lane cycle: $TargetLane"
    Write-Host "- capture_priority: $CapturePriority"
    Write-Host "- freshness_seconds: $FreshnessSeconds"
    Write-Host "- max_events: $MaxEvents"
    Write-Host "- max_markets: $MaxMarkets"
    if ($ModelPath) {
        Write-Host "- model_path: $ModelPath"
    } else {
        Write-Host "- model_path: lane latest_model, then latest_niche_model/latest_model"
    }
    Write-Host "- include_policy_ready_only: $PolicyReadyOnly"

    Invoke-LaneCycleCommand -Command @("lane", "capture-raw") -Arguments (New-CaptureArguments -TargetLane $TargetLane -PolicyReadyOnly $PolicyReadyOnly)
    Write-Host "[phase] refresh lane template before predictions"
    Invoke-LaneCycleCommand -Command @("lane", "run-shadow") -Arguments @("--lane-id", $TargetLane)
    $predictionArguments = @("--lane-id", $TargetLane)
    if ($ModelPath) {
        $predictionArguments += @("--model-path", $ModelPath)
    }
    Invoke-LaneCycleCommand -Command @("lane", "build-predictions") -Arguments $predictionArguments
    Write-Host "[phase] rerun lane with fresh predictions"
    Invoke-LaneCycleCommand -Command @("lane", "run-shadow") -Arguments @("--lane-id", $TargetLane)
    Invoke-LaneCycleCommand -Command @("lane", "report") -Arguments @("--lane-id", $TargetLane)
}

Invoke-MarketLaneCycle -TargetLane $LaneId -PolicyReadyOnly ([bool]$IncludePolicyReadyOnly)

if ($IncludeGoals) {
    Write-Host ""
    Write-Host "Running football_goals_core as research-only lane. It must remain policy_ready=false and can_emit_picks=false."
    Invoke-MarketLaneCycle -TargetLane "football_goals_core" -PolicyReadyOnly $false
}
