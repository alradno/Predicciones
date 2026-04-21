param(
    [string]$RunDir
)

. "$PSScriptRoot\_predicciones.ps1"

$target = if ($RunDir) {
    [pscustomobject]@{
        Command = 'report'
        RunDir = $RunDir
    }
} else {
    Resolve-LatestReportTarget
}

if (-not $target) {
    throw 'No encontre un run reciente en outputs\\runs ni un puntero latest_* util.'
}

Write-Host "Reporte: $($target.Command) -> $($target.RunDir)"

exit (Invoke-PrediccionesCommand -Command $target.Command -Arguments @('--run-dir', $target.RunDir))
