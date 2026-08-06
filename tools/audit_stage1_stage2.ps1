param(
    [string]$InputDir = "new_images",
    [string]$OutDir = "",
    [ValidateSet("debug", "info", "warn", "error")]
    [string]$LogLevel = "info",
    [int]$PreviewCount = 8,
    [switch]$Clean,
    [switch]$SaveModel
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $PSScriptRoot "audit_stage1_stage2.py"

$args = @(
    $scriptPath,
    "--input", $InputDir,
    "--log-level", $LogLevel,
    "--preview-count", $PreviewCount
)

if ($OutDir) {
    $args += @("--out", $OutDir)
}
if ($Clean) {
    $args += "--clean"
}
if ($SaveModel) {
    $args += "--save-model"
}

Push-Location $repoRoot
try {
    & python @args
} finally {
    Pop-Location
}
