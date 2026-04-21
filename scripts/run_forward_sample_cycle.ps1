param(
    [int]$StreamSeconds = 0,
    [int]$MaxCycles = 1,
    [int]$MinStreamSeconds = 60,
    [int]$MaxStreamSeconds = 7200,
    [int]$MarginSeconds = 600,
    [int]$BackoffSeconds = 30,
    [int]$HeartbeatSeconds = 60,
    [int]$CaptureChunkSeconds = 1800,
    [int]$MaxCollectRetries = 3,
    [string]$DbPath = "",
    [string]$PolicyBundle = "",
    [switch]$AllowLongStream,
    [switch]$DryRun
)

. "$PSScriptRoot\_predicciones.ps1"

$repoRoot = Get-PrediccionesRepoRoot
$outputsRoot = Join-Path $repoRoot "outputs"
$defaultDbPath = Join-Path $repoRoot "data\polymarket_shadow.sqlite"
$effectiveDbPath = if ([string]::IsNullOrWhiteSpace($DbPath)) { $defaultDbPath } else { $DbPath }
$effectiveHeartbeatSeconds = [Math]::Max(5, $HeartbeatSeconds)
$effectiveCaptureChunkSeconds = [Math]::Max(60, $CaptureChunkSeconds)
$effectiveMaxCollectRetries = [Math]::Max(0, $MaxCollectRetries)

function Resolve-FrozenPolicyBundle {
    param([string]$ExplicitPolicyBundle)

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPolicyBundle)) {
        $candidate = $ExplicitPolicyBundle
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            $candidate = Join-Path $candidate "policy_bundle.json"
        }
        return $candidate
    }

    $pointer = Join-Path $outputsRoot "latest_polymarket_policy.txt"
    if (-not (Test-Path -LiteralPath $pointer)) {
        return ""
    }
    $candidate = (Get-Content -LiteralPath $pointer -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        return ""
    }
    if (Test-Path -LiteralPath $candidate -PathType Container) {
        $candidate = Join-Path $candidate "policy_bundle.json"
    }
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }
    return ""
}

function Get-PythonExe {
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }
    return "python"
}

function Get-ForwardCapturePlan {
    param(
        [string]$DatabasePath,
        [int]$ExplicitStreamSeconds,
        [int]$MinimumStreamSeconds,
        [int]$MaximumStreamSeconds,
        [int]$StreamMarginSeconds,
        [bool]$AllowLong
    )

    if ($ExplicitStreamSeconds -gt 0) {
        return [pscustomobject]@{
            status = "explicit_stream_seconds"
            stream_seconds = $ExplicitStreamSeconds
            uncapped_stream_seconds = $ExplicitStreamSeconds
            stream_capped = $false
            next_decision_time = ""
            seconds_until_next_decision = $null
            upcoming_windows = @()
        }
    }

    $python = Get-PythonExe
    $env:PRED_FORWARD_DB = $DatabasePath
    $env:PRED_FORWARD_MIN_STREAM = [string]$MinimumStreamSeconds
    $env:PRED_FORWARD_MAX_STREAM = [string]$MaximumStreamSeconds
    $env:PRED_FORWARD_MARGIN = [string]$StreamMarginSeconds
    $env:PRED_FORWARD_ALLOW_LONG = if ($AllowLong) { "1" } else { "0" }
    $code = @'
import json
import math
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

db_path = Path(os.environ["PRED_FORWARD_DB"])
min_stream = int(os.environ["PRED_FORWARD_MIN_STREAM"])
max_stream = int(os.environ["PRED_FORWARD_MAX_STREAM"])
margin = int(os.environ["PRED_FORWARD_MARGIN"])
allow_long = os.environ["PRED_FORWARD_ALLOW_LONG"] == "1"
if not db_path.exists():
    print(json.dumps({"status": "waiting_for_fixtures", "stream_seconds": 0, "uncapped_stream_seconds": 0, "stream_capped": False, "upcoming_windows": []}))
    raise SystemExit(0)

def parse_time(value):
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

now = datetime.now(timezone.utc)
rows = []
with sqlite3.connect(db_path) as con:
    try:
        cursor = con.execute(
            "SELECT group_key, league_code, game_start_time FROM pm_market_groups WHERE mapping_status = 'complete'"
        )
        rows = cursor.fetchall()
    except sqlite3.Error:
        rows = []

upcoming = []
for group_key, league_code, game_start_time in rows:
    kickoff = parse_time(game_start_time)
    if kickoff is None:
        continue
    decision_time = kickoff - timedelta(minutes=45)
    if decision_time < now - timedelta(minutes=5):
        continue
    upcoming.append((decision_time, kickoff, str(group_key), str(league_code)))

upcoming.sort(key=lambda item: item[0])
if not upcoming:
    print(json.dumps({"status": "waiting_for_fixtures", "stream_seconds": 0, "uncapped_stream_seconds": 0, "stream_capped": False, "upcoming_windows": []}))
    raise SystemExit(0)

next_decision, _, _, _ = upcoming[0]
seconds_until = max((next_decision - now).total_seconds(), 0.0)
if seconds_until > 72 * 3600:
    stream_seconds = 0
    uncapped_stream_seconds = 0
    stream_capped = False
    status = "waiting_for_fixtures"
else:
    uncapped_stream_seconds = max(min_stream, int(math.ceil(seconds_until + margin)))
    stream_capped = (not allow_long) and max_stream > 0 and uncapped_stream_seconds > max_stream
    stream_seconds = max_stream if stream_capped else uncapped_stream_seconds
    status = "ready_to_capture"
    if stream_capped:
        status = "ready_to_capture_capped"

print(json.dumps({
    "status": status,
    "stream_seconds": stream_seconds,
    "uncapped_stream_seconds": uncapped_stream_seconds,
    "stream_capped": stream_capped,
    "next_decision_time": next_decision.isoformat(),
    "seconds_until_next_decision": seconds_until,
    "upcoming_windows": [
        {
            "decision_time": item[0].isoformat(),
            "game_start_time": item[1].isoformat(),
            "group_key": item[2],
            "league_code": item[3],
        }
        for item in upcoming[:12]
    ],
}))
'@
    $json = $code | & $python -
    return $json | ConvertFrom-Json
}

function Write-CaptureBlockerReport {
    param(
        [string]$RunDir,
        [object]$ForwardReport,
        [object]$CapturePlan
    )

    if ([string]::IsNullOrWhiteSpace($RunDir) -or -not (Test-Path -LiteralPath $RunDir)) {
        return
    }
    $reportPath = Join-Path $RunDir "capture_blocker_report.json"
    $payload = [ordered]@{
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        sample_status = $ForwardReport.sample_status
        next_action = $ForwardReport.next_action
        collect_plan = $CapturePlan
        blockers = $ForwardReport.cumulative.blockers
        policy_reoptimized = $false
        t45m_policy_touched = $false
    }
    $payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
    Write-Host "- capture_blocker_report: $reportPath"
}

function Write-CollectStreamDiagnostics {
    param([string]$SummaryPath)

    if ([string]::IsNullOrWhiteSpace($SummaryPath) -or -not (Test-Path -LiteralPath $SummaryPath)) {
        return
    }
    $summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    $stream = $summary.stream
    if ($null -eq $stream) {
        return
    }

    function Get-StreamIntValue {
        param([string]$Name)
        $property = $stream.PSObject.Properties[$Name]
        if ($null -eq $property -or $null -eq $property.Value -or $property.Value -eq "") {
            return 0
        }
        return [int]$property.Value
    }

    $marketErrors = Get-StreamIntValue -Name "market_stream_errors"
    $sportsErrors = Get-StreamIntValue -Name "sports_stream_errors"
    $marketCompleted = Get-StreamIntValue -Name "market_stream_completed"
    $sportsCompleted = Get-StreamIntValue -Name "sports_stream_completed"
    Write-Host "- collect_summary: $SummaryPath"
    Write-Host "- websocket_status: market_completed=$marketCompleted market_errors=$marketErrors sports_completed=$sportsCompleted sports_errors=$sportsErrors"
    if ($marketErrors -gt 0 -or $sportsErrors -gt 0) {
        Write-Host "- websocket_degraded: REST checkpoints continued; inspect stream.*_last_error in collect_summary.json"
    }
}

function Get-ForwardDbSnapshot {
    param([string]$DatabasePath)

    $python = Get-PythonExe
    $env:PRED_FORWARD_DB = $DatabasePath
    $code = @'
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

db_path = Path(os.environ["PRED_FORWARD_DB"])
payload = {
    "db_exists": db_path.exists(),
    "book_checkpoints": 0,
    "book_best": 0,
    "trades": 0,
    "shadow_decisions": 0,
    "shadow_fills": 0,
    "market_groups": 0,
    "mapped_groups": 0,
    "decision_checkpoints": 0,
    "periodic_checkpoints": 0,
    "latest_checkpoint_timestamp": "",
    "latest_book_best_timestamp": "",
    "latest_trade_timestamp": "",
    "latest_checkpoint_age_seconds": None,
}
if not db_path.exists():
    print(json.dumps(payload))
    raise SystemExit(0)

def scalar(con, sql):
    try:
        value = con.execute(sql).fetchone()[0]
    except sqlite3.Error:
        return None
    return value

def parse_time(value):
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

with sqlite3.connect(db_path) as con:
    payload["book_checkpoints"] = int(scalar(con, "SELECT COUNT(*) FROM pm_book_checkpoints") or 0)
    payload["book_best"] = int(scalar(con, "SELECT COUNT(*) FROM pm_book_best") or 0)
    payload["trades"] = int(scalar(con, "SELECT COUNT(*) FROM pm_trades") or 0)
    payload["shadow_decisions"] = int(scalar(con, "SELECT COUNT(*) FROM pm_shadow_decisions") or 0)
    payload["shadow_fills"] = int(scalar(con, "SELECT COUNT(*) FROM pm_shadow_fills") or 0)
    payload["market_groups"] = int(scalar(con, "SELECT COUNT(*) FROM pm_market_groups") or 0)
    payload["mapped_groups"] = int(scalar(con, "SELECT COUNT(*) FROM pm_market_groups WHERE mapping_status = 'complete'") or 0)
    payload["decision_checkpoints"] = int(scalar(con, "SELECT COUNT(*) FROM pm_book_checkpoints WHERE event_type LIKE 'decision%'") or 0)
    payload["periodic_checkpoints"] = int(scalar(con, "SELECT COUNT(*) FROM pm_book_checkpoints WHERE event_type = 'periodic_checkpoint'") or 0)
    payload["latest_checkpoint_timestamp"] = str(scalar(con, "SELECT MAX(timestamp) FROM pm_book_checkpoints") or "")
    payload["latest_book_best_timestamp"] = str(scalar(con, "SELECT MAX(timestamp) FROM pm_book_best") or "")
    payload["latest_trade_timestamp"] = str(scalar(con, "SELECT MAX(timestamp) FROM pm_trades") or "")

latest = parse_time(payload["latest_checkpoint_timestamp"])
if latest is not None:
    payload["latest_checkpoint_age_seconds"] = max(0, int((datetime.now(timezone.utc) - latest).total_seconds()))

print(json.dumps(payload))
'@
    try {
        $json = $code | & $python -
        return $json | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{
            db_exists = $false
            book_checkpoints = 0
            book_best = 0
            trades = 0
            shadow_decisions = 0
            shadow_fills = 0
            market_groups = 0
            mapped_groups = 0
            decision_checkpoints = 0
            periodic_checkpoints = 0
            latest_checkpoint_timestamp = ""
            latest_book_best_timestamp = ""
            latest_trade_timestamp = ""
            latest_checkpoint_age_seconds = $null
        }
    }
}

function Get-SnapshotInt {
    param(
        [object]$Snapshot,
        [string]$Name
    )

    if ($null -eq $Snapshot) {
        return 0
    }
    $property = $Snapshot.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value -or $property.Value -eq "") {
        return 0
    }
    return [int64]$property.Value
}

function Get-SnapshotText {
    param(
        [object]$Snapshot,
        [string]$Name
    )

    if ($null -eq $Snapshot) {
        return ""
    }
    $property = $Snapshot.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return ""
    }
    return [string]$property.Value
}

function Write-ForwardDbHeartbeat {
    param(
        [string]$Phase,
        [object]$StartSnapshot,
        [object]$PreviousSnapshot,
        [object]$CurrentSnapshot,
        [datetime]$StartedAt
    )

    $elapsedSeconds = [int]([datetime]::Now - $StartedAt).TotalSeconds
    $checkpointNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "book_checkpoints"
    $checkpointStart = Get-SnapshotInt -Snapshot $StartSnapshot -Name "book_checkpoints"
    $checkpointPrevious = Get-SnapshotInt -Snapshot $PreviousSnapshot -Name "book_checkpoints"
    $bookBestNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "book_best"
    $bookBestStart = Get-SnapshotInt -Snapshot $StartSnapshot -Name "book_best"
    $bookBestPrevious = Get-SnapshotInt -Snapshot $PreviousSnapshot -Name "book_best"
    $tradesNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "trades"
    $tradesStart = Get-SnapshotInt -Snapshot $StartSnapshot -Name "trades"
    $tradesPrevious = Get-SnapshotInt -Snapshot $PreviousSnapshot -Name "trades"
    $decisionsNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "shadow_decisions"
    $decisionsStart = Get-SnapshotInt -Snapshot $StartSnapshot -Name "shadow_decisions"
    $fillsNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "shadow_fills"
    $fillsStart = Get-SnapshotInt -Snapshot $StartSnapshot -Name "shadow_fills"
    $groupsNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "market_groups"
    $mappedNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "mapped_groups"
    $decisionCheckpointsNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "decision_checkpoints"
    $periodicCheckpointsNow = Get-SnapshotInt -Snapshot $CurrentSnapshot -Name "periodic_checkpoints"
    $intervalDelta = ($checkpointNow - $checkpointPrevious) + ($bookBestNow - $bookBestPrevious) + ($tradesNow - $tradesPrevious)
    $activity = if ($intervalDelta -gt 0) { "capturing_new_rows" } else { "alive_no_new_rows_yet" }
    $latestCheckpoint = Get-SnapshotText -Snapshot $CurrentSnapshot -Name "latest_checkpoint_timestamp"
    if ([string]::IsNullOrWhiteSpace($latestCheckpoint)) {
        $latestCheckpoint = "none"
    }
    $checkpointAge = Get-SnapshotText -Snapshot $CurrentSnapshot -Name "latest_checkpoint_age_seconds"
    if ([string]::IsNullOrWhiteSpace($checkpointAge)) {
        $checkpointAge = "n/a"
    }

    Write-Host (
        "[heartbeat] phase={0} elapsed={1}s status={2} checkpoints={3} (+{4} cycle,+{5} interval) best={6} (+{7} cycle,+{8} interval) trades={9} (+{10} cycle,+{11} interval) decisions={12} (+{13}) fills={14} (+{15}) groups={16}/{17} decision_ckpt={18} periodic_ckpt={19} latest_checkpoint={20} age={21}s" -f `
            $Phase,
            $elapsedSeconds,
            $activity,
            $checkpointNow,
            ($checkpointNow - $checkpointStart),
            ($checkpointNow - $checkpointPrevious),
            $bookBestNow,
            ($bookBestNow - $bookBestStart),
            ($bookBestNow - $bookBestPrevious),
            $tradesNow,
            ($tradesNow - $tradesStart),
            ($tradesNow - $tradesPrevious),
            $decisionsNow,
            ($decisionsNow - $decisionsStart),
            $fillsNow,
            ($fillsNow - $fillsStart),
            $mappedNow,
            $groupsNow,
            $decisionCheckpointsNow,
            $periodicCheckpointsNow,
            $latestCheckpoint,
            $checkpointAge
    )
}

function Invoke-ForwardCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

    [string[]]$Arguments = @()
    )

    $venvExe = Join-Path $repoRoot ".venv\Scripts\predicciones.exe"
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

    if (Test-Path -LiteralPath $venvExe) {
        & $venvExe $Command @Arguments
        return $LASTEXITCODE
    }

    if (Test-Path -LiteralPath $venvPython) {
        $srcPath = Join-Path $repoRoot "src"
        if ($env:PYTHONPATH) {
            $env:PYTHONPATH = "$srcPath$([IO.Path]::PathSeparator)$env:PYTHONPATH"
        } else {
            $env:PYTHONPATH = $srcPath
        }
        & $venvPython -m predicciones.cli $Command @Arguments
        return $LASTEXITCODE
    }

    & predicciones $Command @Arguments
    return $LASTEXITCODE
}

function Invoke-ForwardCommandWithHeartbeat {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string[]]$Arguments = @(),
        [string]$Phase = $Command,
        [string]$DatabasePath = $effectiveDbPath,
        [int]$EverySeconds = 60
    )

    $sleepSeconds = [Math]::Max(5, $EverySeconds)
    $startedAt = Get-Date
    $startSnapshot = Get-ForwardDbSnapshot -DatabasePath $DatabasePath
    $previousSnapshot = $startSnapshot
    Write-ForwardDbHeartbeat -Phase $Phase -StartSnapshot $startSnapshot -PreviousSnapshot $startSnapshot -CurrentSnapshot $startSnapshot -StartedAt $startedAt

    $job = Start-Job -ScriptBlock {
        param($JobRepoRoot, $JobCommand, $JobArguments)

        Set-Location -LiteralPath $JobRepoRoot
        $venvExe = Join-Path $JobRepoRoot ".venv\Scripts\predicciones.exe"
        $venvPython = Join-Path $JobRepoRoot ".venv\Scripts\python.exe"
        $exitCode = 0

        if (Test-Path -LiteralPath $venvExe) {
            & $venvExe $JobCommand @JobArguments
            $exitCode = $LASTEXITCODE
        } elseif (Test-Path -LiteralPath $venvPython) {
            $srcPath = Join-Path $JobRepoRoot "src"
            if ($env:PYTHONPATH) {
                $env:PYTHONPATH = "$srcPath$([IO.Path]::PathSeparator)$env:PYTHONPATH"
            } else {
                $env:PYTHONPATH = $srcPath
            }
            & $venvPython -m predicciones.cli $JobCommand @JobArguments
            $exitCode = $LASTEXITCODE
        } else {
            & predicciones $JobCommand @JobArguments
            $exitCode = $LASTEXITCODE
        }

        if ($null -eq $exitCode) {
            $exitCode = 0
        }
        Write-Output "__PREDICCIONES_EXIT_CODE__=$exitCode"
        exit $exitCode
    } -ArgumentList $repoRoot, $Command, $Arguments

    $exitCode = $null
    while ($job.State -eq "Running") {
        Start-Sleep -Seconds $sleepSeconds
        $lines = Receive-Job -Job $job
        foreach ($line in $lines) {
            $text = [string]$line
            if ($text.StartsWith("__PREDICCIONES_EXIT_CODE__=")) {
                $exitCode = [int]($text.Substring("__PREDICCIONES_EXIT_CODE__=".Length))
            } elseif (-not [string]::IsNullOrWhiteSpace($text)) {
                Write-Host $text
            }
        }
        $currentSnapshot = Get-ForwardDbSnapshot -DatabasePath $DatabasePath
        Write-ForwardDbHeartbeat -Phase $Phase -StartSnapshot $startSnapshot -PreviousSnapshot $previousSnapshot -CurrentSnapshot $currentSnapshot -StartedAt $startedAt
        $previousSnapshot = $currentSnapshot
    }

    $lines = Receive-Job -Job $job
    foreach ($line in $lines) {
        $text = [string]$line
        if ($text.StartsWith("__PREDICCIONES_EXIT_CODE__=")) {
            $exitCode = [int]($text.Substring("__PREDICCIONES_EXIT_CODE__=".Length))
        } elseif (-not [string]::IsNullOrWhiteSpace($text)) {
            Write-Host $text
        }
    }
    $finalSnapshot = Get-ForwardDbSnapshot -DatabasePath $DatabasePath
    Write-ForwardDbHeartbeat -Phase "$Phase complete" -StartSnapshot $startSnapshot -PreviousSnapshot $previousSnapshot -CurrentSnapshot $finalSnapshot -StartedAt $startedAt

    if ($null -eq $exitCode) {
        $exitCode = if ($job.State -eq "Failed") { 1 } else { 0 }
    }
    Remove-Job -Job $job -Force
    return [int]$exitCode
}

function Invoke-CollectPolymarketSafely {
    param(
        [int]$TotalStreamSeconds,
        [string]$DatabasePath,
        [int]$ChunkSeconds,
        [int]$HeartbeatEverySeconds,
        [int]$RetryLimit,
        [int]$RetryBackoffSeconds
    )

    $totalSeconds = [Math]::Max(0, $TotalStreamSeconds)
    if ($totalSeconds -le 0) {
        $plannedChunks = @([pscustomobject]@{ index = 1; total = 1; seconds = 0; remaining_after = 0 })
    } else {
        $chunks = New-Object System.Collections.Generic.List[object]
        $remaining = $totalSeconds
        $index = 1
        $total = [int][Math]::Ceiling($totalSeconds / [double]$ChunkSeconds)
        while ($remaining -gt 0) {
            $seconds = [Math]::Min($ChunkSeconds, $remaining)
            $remainingAfter = [Math]::Max(0, $remaining - $seconds)
            $chunks.Add([pscustomobject]@{ index = $index; total = $total; seconds = $seconds; remaining_after = $remainingAfter })
            $remaining = $remainingAfter
            $index += 1
        }
        $plannedChunks = $chunks.ToArray()
    }

    Write-Host "- collect_chunk_seconds: $ChunkSeconds"
    Write-Host "- collect_chunks_planned: $($plannedChunks.Count)"
    Write-Host "- collect_retry_limit_per_chunk: $RetryLimit"

    $completedChunks = 0
    $failedChunks = 0
    $lastExitCode = 0

    foreach ($chunk in $plannedChunks) {
        $attempt = 0
        $chunkSucceeded = $false
        while (-not $chunkSucceeded -and $attempt -le $RetryLimit) {
            $attempt += 1
            $collectArgs = @("--db-path", $DatabasePath, "--stream-seconds", [string]$chunk.seconds)
            $retryLabel = if ($attempt -eq 1) { "initial" } else { "retry_$($attempt - 1)" }
            Write-Host "[collect-chunk] chunk=$($chunk.index)/$($chunk.total) attempt=$retryLabel stream_seconds=$($chunk.seconds) remaining_after=$($chunk.remaining_after)"
            $exitCode = Invoke-ForwardCommandWithHeartbeat `
                -Command "collect-polymarket" `
                -Arguments $collectArgs `
                -Phase "collect-polymarket chunk $($chunk.index)/$($chunk.total)" `
                -DatabasePath $DatabasePath `
                -EverySeconds $HeartbeatEverySeconds
            $lastExitCode = $exitCode
            $collectSummaryPath = Get-LatestRunPointer -PointerName "latest_polymarket_collect.txt"
            Write-CollectStreamDiagnostics -SummaryPath $collectSummaryPath
            if ($exitCode -eq 0) {
                $chunkSucceeded = $true
                $completedChunks += 1
                Write-Host "[collect-chunk] chunk=$($chunk.index)/$($chunk.total) status=completed"
                continue
            }

            Write-Warning "collect-polymarket chunk $($chunk.index)/$($chunk.total) failed with exit code $exitCode"
            if ($attempt -le $RetryLimit) {
                Write-Host "[collect-chunk] retrying_after=${RetryBackoffSeconds}s"
                Start-Sleep -Seconds $RetryBackoffSeconds
            }
        }

        if (-not $chunkSucceeded) {
            $failedChunks += 1
            Write-Warning "collect-polymarket chunk $($chunk.index)/$($chunk.total) exhausted retries. Progress already written to SQLite will be kept."
            break
        }
    }

    $status = if ($failedChunks -eq 0) {
        "completed"
    } elseif ($completedChunks -gt 0) {
        "partial_collect_failed"
    } else {
        "failed"
    }
    $exitCode = if ($status -eq "failed") { $lastExitCode } else { 0 }
    return [pscustomobject]@{
        status = $status
        exit_code = [int]$exitCode
        completed_chunks = [int]$completedChunks
        failed_chunks = [int]$failedChunks
        planned_chunks = [int]$plannedChunks.Count
    }
}

$frozenPolicyBundle = Resolve-FrozenPolicyBundle -ExplicitPolicyBundle $PolicyBundle
$policyArgs = @()
if (-not [string]::IsNullOrWhiteSpace($frozenPolicyBundle)) {
    $policyArgs = @("--policy-bundle", $frozenPolicyBundle)
}

for ($cycle = 1; $cycle -le [Math]::Max(1, $MaxCycles); $cycle++) {
    Write-Host "Forward sample cycle $cycle/$MaxCycles"
    Write-Host "[phase 1/4] planning_capture"

    $capturePlan = Get-ForwardCapturePlan `
        -DatabasePath $effectiveDbPath `
        -ExplicitStreamSeconds $StreamSeconds `
        -MinimumStreamSeconds $MinStreamSeconds `
        -MaximumStreamSeconds $MaxStreamSeconds `
        -StreamMarginSeconds $MarginSeconds `
        -AllowLong ([bool]$AllowLongStream)

    $shadowArgs = @("--db-path", $effectiveDbPath) + $policyArgs

    Write-Host "- capture_status: $($capturePlan.status)"
    Write-Host "- stream_seconds: $($capturePlan.stream_seconds)"
    Write-Host "- heartbeat_seconds: $effectiveHeartbeatSeconds"
    Write-Host "- capture_chunk_seconds: $effectiveCaptureChunkSeconds"
    Write-Host "- max_collect_retries: $effectiveMaxCollectRetries"
    if ($capturePlan.next_decision_time) {
        Write-Host "- next_decision_time: $($capturePlan.next_decision_time)"
    }
    if ($capturePlan.stream_capped) {
        Write-Host "- uncapped_stream_seconds: $($capturePlan.uncapped_stream_seconds)"
        Write-Host "- stream_cap: capped to $MaxStreamSeconds seconds; use -AllowLongStream to cover the whole window across resumable chunks"
    }
    if ($frozenPolicyBundle) {
        Write-Host "- frozen_policy_bundle: $frozenPolicyBundle"
    } else {
        Write-Host "- frozen_policy_bundle: model_bundle_fallback"
    }

    if ($DryRun) {
        $dryRunChunkCount = if ([int]$capturePlan.stream_seconds -le 0) { 1 } else { [int][Math]::Ceiling([int]$capturePlan.stream_seconds / [double]$effectiveCaptureChunkSeconds) }
        Write-Host "[dry-run] phase=planning_capture complete"
        Write-Host "[dry-run] collect_chunks_planned=$dryRunChunkCount"
        Write-Host "[dry-run] each chunk runs: predicciones collect-polymarket --db-path $effectiveDbPath --stream-seconds <chunk_seconds>"
        Write-Host "[dry-run] predicciones shadow-polymarket $($shadowArgs -join ' ')"
        continue
    }

    Write-Host "[phase 2/4] collect-polymarket"
    if ([int]$capturePlan.stream_seconds -gt 60) {
        $minutes = [Math]::Round([int]$capturePlan.stream_seconds / 60.0, 1)
        Write-Host "- collecting_books: total target ~$minutes minutes, split into resumable chunks with heartbeat every $effectiveHeartbeatSeconds seconds"
    }
    $collectResult = Invoke-CollectPolymarketSafely `
        -TotalStreamSeconds ([int]$capturePlan.stream_seconds) `
        -DatabasePath $effectiveDbPath `
        -ChunkSeconds $effectiveCaptureChunkSeconds `
        -HeartbeatEverySeconds $effectiveHeartbeatSeconds `
        -RetryLimit $effectiveMaxCollectRetries `
        -RetryBackoffSeconds $BackoffSeconds
    Write-Host "- collect_status: $($collectResult.status)"
    Write-Host "- collect_chunks_completed: $($collectResult.completed_chunks)/$($collectResult.planned_chunks)"
    if ([int]$collectResult.exit_code -ne 0) {
        Write-Warning "collect-polymarket failed before any durable chunk completed with exit code $($collectResult.exit_code)"
        if ($cycle -lt $MaxCycles) {
            Start-Sleep -Seconds $BackoffSeconds
            continue
        }
        exit ([int]$collectResult.exit_code)
    }
    if ($collectResult.status -eq "partial_collect_failed") {
        Write-Warning "Continuing to shadow/report with partial captured data. Rerunning the same command later resumes from SQLite."
    }

    Write-Host "[phase 3/4] shadow-polymarket"
    $shadowStartedAt = Get-Date
    $shadowStartSnapshot = Get-ForwardDbSnapshot -DatabasePath $effectiveDbPath
    Write-ForwardDbHeartbeat -Phase "shadow-polymarket start" -StartSnapshot $shadowStartSnapshot -PreviousSnapshot $shadowStartSnapshot -CurrentSnapshot $shadowStartSnapshot -StartedAt $shadowStartedAt
    $shadowExit = Invoke-ForwardCommand -Command "shadow-polymarket" -Arguments $shadowArgs
    $shadowFinalSnapshot = Get-ForwardDbSnapshot -DatabasePath $effectiveDbPath
    Write-ForwardDbHeartbeat -Phase "shadow-polymarket complete" -StartSnapshot $shadowStartSnapshot -PreviousSnapshot $shadowStartSnapshot -CurrentSnapshot $shadowFinalSnapshot -StartedAt $shadowStartedAt
    if ($shadowExit -ne 0) {
        Write-Warning "shadow-polymarket failed with exit code $shadowExit"
        if ($cycle -lt $MaxCycles) {
            Start-Sleep -Seconds $BackoffSeconds
            continue
        }
        exit $shadowExit
    }

    $shadowRun = Get-LatestRunPointer -PointerName "latest_polymarket_shadow.txt"
    if (-not $shadowRun) {
        $shadowRun = Get-LatestRunDirectory
    }
    if ($shadowRun) {
        Write-Host "[phase 4/4] report-polymarket"
        Invoke-ForwardCommand -Command "report-polymarket" -Arguments @("--run-dir", $shadowRun) | Out-Host
        $forwardReportPath = Join-Path $shadowRun "forward_sample_report.json"
        if (Test-Path -LiteralPath $forwardReportPath) {
            $forwardReport = Get-Content -LiteralPath $forwardReportPath -Raw | ConvertFrom-Json
            Write-Host "- forward_sample_status: $($forwardReport.sample_status)"
            Write-Host "- forward_next_action: $($forwardReport.next_action)"
            if ($forwardReport.sample_status -eq "sample_ready") {
                Write-Host "- sample_ready: forward decision-region analysis can start from this ledger."
                break
            }
            if ($forwardReport.sample_status -eq "coverage_blocked") {
                Write-CaptureBlockerReport -RunDir $shadowRun -ForwardReport $forwardReport -CapturePlan $capturePlan
            }
            if ($forwardReport.sample_status -eq "settlement_pending") {
                Write-Host "- settlement_pending: next cycle will refresh resolutions without changing policy."
            }
        }
    }

    if ($cycle -lt $MaxCycles) {
        Start-Sleep -Seconds $BackoffSeconds
    }
}
