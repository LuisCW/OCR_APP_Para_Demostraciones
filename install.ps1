# Instalador del Escaner OCR con LaTeX
Write-Host "======================================"
Write-Host "Instalador del Escaner OCR con LaTeX"
Write-Host "======================================"
Write-Host ""

# Verificar Python
Write-Host "Verificando Python..."
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python encontrado: $pythonVersion"
} catch {
    Write-Host "ERROR: Python no esta instalado o no esta en el PATH" -ForegroundColor Red
    Write-Host "Por favor instala Python desde https://python.org"
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host ""

# Crear entorno virtual (opcional)
Write-Host "¿Deseas crear un entorno virtual? (recomendado) [Y/n]:"
$createVenv = Read-Host
if ($createVenv -eq "" -or $createVenv.ToLower() -eq "y") {
    Write-Host "Creando entorno virtual..."
    python -m venv venv
    
    # Activar entorno virtual
    Write-Host "Activando entorno virtual..."
    & ".\venv\Scripts\Activate.ps1"
}

Write-Host ""
Write-Host "Instalando dependencias..."

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

Write-Host ""
Write-Host "======================================"
Write-Host "Instalacion completada!" -ForegroundColor Green
Write-Host "======================================"
Write-Host ""

Write-Host "NOTA IMPORTANTE:" -ForegroundColor Yellow
Write-Host "Para usar Tesseract OCR, necesitas instalar Tesseract por separado:"
Write-Host "1. Descarga Tesseract desde: https://github.com/UB-Mannheim/tesseract/wiki"
Write-Host "2. Instala y agrega al PATH del sistema"
Write-Host "3. O configura la variable TESSERACT_CMD en el codigo"
Write-Host ""

Write-Host "Para ejecutar la aplicacion, usa:"
Write-Host "python main.py" -ForegroundColor Cyan
Write-Host ""

Read-Host "Presiona Enter para salir"
