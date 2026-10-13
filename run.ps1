Get-ChildItem -Filter "*.py" | ForEach-Object {
    python $_.FullName
}
