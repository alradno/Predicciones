param(
    [int]$Days = 30,
    [switch]$Delete
)

. "$PSScriptRoot\_predicciones.ps1"

$protected = @(
    (Get-LatestRunPointer -PointerName 'latest_backtest.txt'),
    (Get-LatestRunPointer -PointerName 'latest_backtest_net.txt'),
    (Get-LatestRunPointer -PointerName 'latest_polymarket_shadow.txt'),
    (Get-LatestRunPointer -PointerName 'latest_polymarket_retro.txt')
)

$candidates = Get-OldRunDirectories -Days $Days -ProtectedRunDirs $protected

if (-not $candidates -or $candidates.Count -eq 0) {
    Write-Host "No encontre runs antiguos para limpiar en outputs\\runs."
    exit 0
}

Write-Host "Runs candidatos para limpiar (mas viejos que $Days dias):"
$candidates | Select-Object LastWriteTime, FullName | Format-Table -AutoSize

if (-not $Delete) {
    Write-Host 'Vista previa solamente. Pasa -Delete para borrar estos directorios.'
    exit 0
}

foreach ($candidate in $candidates) {
    Remove-Item -LiteralPath $candidate.FullName -Recurse -Force
    Write-Host "Borrado: $($candidate.FullName)"
}
