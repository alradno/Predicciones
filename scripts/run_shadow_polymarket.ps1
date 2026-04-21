param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

. "$PSScriptRoot\_predicciones.ps1"

exit (Invoke-PrediccionesCommand -Command 'shadow-polymarket' -Arguments $RemainingArgs)
