# Скрипт для синхронизации с Kaggle
# Требует установленного Kaggle CLI

if (!(Test-Path "kaggle_upload\dataset-metadata.json")) {
    Write-Error "Сначала запустите: python scripts/prepare_kaggle_data.py"
    exit
}

$metadata = Get-Content "kaggle_upload\dataset-metadata.json" | ConvertFrom-Json
$id = $metadata.id

if ($id -eq "your-username/bebladii-resources") {
    Write-Warning "ВНИМАНИЕ: Измените 'id' в kaggle_upload\dataset-metadata.json на ваш реальный username/название!"
    exit
}

Write-Host "Попытка обновления существующего датасета: $id..."
kaggle datasets version -p kaggle_upload -m "Auto-update $(Get-Date -Format 'yyyy-MM-dd HH:mm')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Датасет не существует или ошибка. Пробую создать новый..."
    kaggle datasets create -p kaggle_upload
}
