function Get-PrediccionesRepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

function Invoke-PrediccionesCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Command,

        [string[]]$Arguments = @()
    )

    $repoRoot = Get-PrediccionesRepoRoot
    $commandParts = @($Command) + @($Arguments)
    $venvExe = Join-Path $repoRoot '.venv\Scripts\predicciones.exe'
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'

    if (Test-Path -LiteralPath $venvExe) {
        & $venvExe @commandParts
        return $LASTEXITCODE
    }

    if (Test-Path -LiteralPath $venvPython) {
        $srcPath = Join-Path $repoRoot 'src'
        if ($env:PYTHONPATH) {
            $env:PYTHONPATH = "$srcPath$([IO.Path]::PathSeparator)$env:PYTHONPATH"
        } else {
            $env:PYTHONPATH = $srcPath
        }

        & $venvPython -m predicciones.cli @commandParts
        return $LASTEXITCODE
    }

    & predicciones @commandParts
    return $LASTEXITCODE
}
