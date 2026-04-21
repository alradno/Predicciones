param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

. "$PSScriptRoot\_predicciones.ps1"

exit (Invoke-PrediccionesCommand -Command 'backtest-net' -Arguments $RemainingArgs)
