@echo off
echo ======================================
echo Instalador del Escaner OCR con LaTeX
echo ======================================
echo.

echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo Por favor instala Python desde https://python.org
    pause
    exit /b 1
)

echo Python encontrado.
echo.

echo Instalando dependencias...
echo.

pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ======================================
echo Instalacion completada!
echo ======================================
echo.

echo NOTA IMPORTANTE:
echo Para usar Tesseract OCR, necesitas instalar Tesseract por separado:
echo 1. Descarga Tesseract desde: https://github.com/UB-Mannheim/tesseract/wiki
echo 2. Instala y agrega al PATH del sistema
echo 3. O configura la variable TESSERACT_CMD en el codigo
echo.

echo Para ejecutar la aplicacion, usa:
echo python main.py
echo.

pause
