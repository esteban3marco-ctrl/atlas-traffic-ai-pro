# Configuration for Local Claude Code via Ollama
$env:ANTHROPIC_BASE_URL = "http://localhost:11434/v1"
$env:ANTHROPIC_API_KEY = "ollama"
$env:ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

Write-Host "🚀 Iniciando Claude Code en modo LOCAL (Ollama)..." -ForegroundColor Cyan
Write-Host "Ubicación del proyecto: $PWD" -ForegroundColor Gray
Write-Host "Habilidades listas: /plugin, /config, /model" -ForegroundColor Gray

# Launch Claude Code
& 'C:\Users\User\AppData\Roaming\npm\claude.cmd'
