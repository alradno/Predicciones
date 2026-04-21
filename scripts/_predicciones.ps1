function Get-PrediccionesRepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

function Invoke-PrediccionesCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string[]]$Arguments = @()
    )

    $repoRoot = Get-PrediccionesRepoRoot
    $venvExe = Join-Path $repoRoot '.venv\Scripts\predicciones.exe'
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'

    if (Test-Path -LiteralPath $venvExe) {
        & $venvExe $Command @Arguments
        return $LASTEXITCODE
    }

    if (Test-Path -LiteralPath $venvPython) {
        $srcPath = Join-Path $repoRoot 'src'
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

function Get-LatestRunPointer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PointerName
    )

    $pointerPath = Join-Path (Join-Path (Get-PrediccionesRepoRoot) 'outputs') $PointerName
    if (-not (Test-Path -LiteralPath $pointerPath)) {
        return $null
    }

    $value = (Get-Content -LiteralPath $pointerPath -Raw).Trim()
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $null
    }

    return $value
}

function Get-LatestRunDirectory {
    $runsRoot = Join-Path (Join-Path (Get-PrediccionesRepoRoot) 'outputs') 'runs'
    if (-not (Test-Path -LiteralPath $runsRoot)) {
        return $null
    }

    $latestRun = Get-ChildItem -LiteralPath $runsRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $latestRun) {
        return $null
    }

    return $latestRun.FullName
}

function Resolve-LatestReportTarget {
    $pointerMap = @(
        @{ Pointer = 'latest_backtest.txt'; Command = 'report' },
        @{ Pointer = 'latest_backtest_net.txt'; Command = 'promote-report' },
        @{ Pointer = 'latest_polymarket_shadow.txt'; Command = 'report-polymarket' },
        @{ Pointer = 'latest_polymarket_retro.txt'; Command = 'report-polymarket-retro' }
    )

    $latestPointer = $null
    foreach ($entry in $pointerMap) {
        $pointerPath = Join-Path (Join-Path (Get-PrediccionesRepoRoot) 'outputs') $entry.Pointer
        if (-not (Test-Path -LiteralPath $pointerPath)) {
            continue
        }

        $runDir = Get-LatestRunPointer -PointerName $entry.Pointer
        if (-not $runDir) {
            continue
        }

        $pointerInfo = [pscustomobject]@{
            Command = $entry.Command
            RunDir = $runDir
            Source = $entry.Pointer
            LastWriteTime = (Get-Item -LiteralPath $pointerPath).LastWriteTime
        }

        if ($null -eq $latestPointer -or $pointerInfo.LastWriteTime -gt $latestPointer.LastWriteTime) {
            $latestPointer = $pointerInfo
        }
    }

    if ($latestPointer) {
        return [pscustomobject]@{
            Command = $latestPointer.Command
            RunDir = $latestPointer.RunDir
            Source = $latestPointer.Source
            LastWriteTime = $latestPointer.LastWriteTime
        }
    }

    $latestRun = Get-LatestRunDirectory
    if (-not $latestRun) {
        return $null
    }

    switch -regex ([IO.Path]::GetFileName($latestRun)) {
        '^backtest_net_' {
            return [pscustomobject]@{ Command = 'promote-report'; RunDir = $latestRun; Source = 'outputs\runs' }
        }
        '^shadow_polymarket_' {
            return [pscustomobject]@{ Command = 'report-polymarket'; RunDir = $latestRun; Source = 'outputs\runs' }
        }
        '^backtest_polymarket_retro_' {
            return [pscustomobject]@{ Command = 'report-polymarket-retro'; RunDir = $latestRun; Source = 'outputs\runs' }
        }
        default {
            return [pscustomobject]@{ Command = 'report'; RunDir = $latestRun; Source = 'outputs\runs' }
        }
    }
}

function Get-OldRunDirectories {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Days,

        [string[]]$ProtectedRunDirs = @()
    )

    $runsRoot = Join-Path (Join-Path (Get-PrediccionesRepoRoot) 'outputs') 'runs'
    if (-not (Test-Path -LiteralPath $runsRoot)) {
        return @()
    }

    $cutoff = (Get-Date).AddDays(-$Days)
    $protected = @{}
    foreach ($path in $ProtectedRunDirs) {
        if ($path) {
            $resolved = Resolve-Path -LiteralPath $path -ErrorAction SilentlyContinue
            if ($resolved) {
                $protected[$resolved.Path] = $true
            }
        }
    }

    return Get-ChildItem -LiteralPath $runsRoot -Directory |
        Where-Object { $_.LastWriteTime -lt $cutoff -and -not $protected.ContainsKey($_.FullName) } |
        Sort-Object LastWriteTime
}
