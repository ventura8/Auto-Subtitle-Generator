# Run tests with coverage and generate the badge
# This script ensures that coverage.xml is generated and the badge is updated.

Write-Host "Running tests with coverage..." -ForegroundColor Cyan
pytest --cov=auto_subtitle --cov-branch --cov-report=xml --cov-report=term

if ($LASTEXITCODE -eq 0) {
    Write-Host "Tests passed! Generating coverage report..." -ForegroundColor Green
}
else {
    Write-Host "Tests failed or coverage threshold not met, but attempting to update badge anyway..." -ForegroundColor Yellow
}

# The badge is actually updated automatically by tests/conftest.py via pytest_sessionfinish.
# However, we can also run it manually if needed:
# python tests/transform_coverage.py coverage.xml

Write-Host "Done." -ForegroundColor Gray
