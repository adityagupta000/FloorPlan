# COMMIT 6: Remaining files
Write-Host "Commit 6/6: Project structure" -ForegroundColor Cyan
git add . 2>$null

$status = git status --short
if ($status) {
    git commit -m "chore: Add remaining project structure

- Empty directories with .gitkeep
- Additional configuration files
- Project scaffolding" 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Remaining files committed`n" -ForegroundColor Green
    }
} else {
    Write-Host "  No remaining files to commit`n" -ForegroundColor Yellow
}