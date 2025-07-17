@echo off
cls
echo =========================================
echo   SOLUCION DIRECTA PARA TESSERACT
echo   Especial para: "B contiene AuB"
echo =========================================
echo.

echo [1/4] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no encontrado. Por favor instala Python primero.
    pause
    exit /b 1
)
echo ✅ Python OK

echo.
echo [2/4] Instalando dependencias Python...
pip install --upgrade pytesseract easyocr opencv-python Pillow numpy matplotlib PyQt5

echo.
echo [3/4] Verificando Tesseract...
tesseract --version >nul 2>&1
if not errorlevel 1 (
    echo ✅ Tesseract ya está disponible en PATH
    goto :test_app
)

echo ❌ Tesseract no está en PATH. Buscando instalación...

rem Buscar Tesseract en ubicaciones comunes
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo ✅ Tesseract encontrado en Program Files
    set "PATH=%PATH%;C:\Program Files\Tesseract-OCR"
    goto :test_app
)

if exist "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" (
    echo ✅ Tesseract encontrado en Program Files (x86)
    set "PATH=%PATH%;C:\Program Files (x86)\Tesseract-OCR"
    goto :test_app
)

echo.
echo ❌ Tesseract no encontrado. INSTALANDO AUTOMATICAMENTE...
echo.

rem Crear directorio temporal
mkdir "%TEMP%\tesseract_install" 2>nul
cd /d "%TEMP%\tesseract_install"

echo Descargando Tesseract OCR...
powershell -Command "& {Invoke-WebRequest -Uri 'https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe' -OutFile 'tesseract-installer.exe'}"

if not exist "tesseract-installer.exe" (
    echo ❌ Error descargando Tesseract.
    echo.
    echo INSTALACION MANUAL REQUERIDA:
    echo 1. Ir a: https://github.com/UB-Mannheim/tesseract/wiki
    echo 2. Descargar: tesseract-ocr-w64-setup-v5.3.0.20221214.exe
    echo 3. Ejecutar como administrador
    echo 4. Instalar en: C:\Program Files\Tesseract-OCR
    echo 5. Marcar "Add to PATH"
    echo 6. Reiniciar PowerShell
    pause
    exit /b 1
)

echo Instalando Tesseract...
echo IMPORTANTE: Durante la instalacion marca "Add to PATH"
tesseract-installer.exe /S /D="C:\Program Files\Tesseract-OCR"

echo Esperando instalacion...
timeout /t 10 /nobreak >nul

rem Verificar instalación
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo ✅ Tesseract instalado correctamente
    set "PATH=%PATH%;C:\Program Files\Tesseract-OCR"
    
    rem Limpiar archivos temporales
    cd /d "%TEMP%"
    rmdir /s /q "tesseract_install" 2>nul
) else (
    echo ❌ Instalacion falló. Prueba instalacion manual.
    pause
    exit /b 1
)

:test_app
echo.
echo [4/4] Probando configuración OCR...

python -c "
import sys
import os

print('🔍 VERIFICANDO CONFIGURACION OCR...')
print('='*50)

# Tesseract
try:
    import pytesseract
    
    # Buscar Tesseract
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    
    tesseract_found = False
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f'✅ Tesseract configurado: {path}')
            
            try:
                version = pytesseract.get_tesseract_version()
                print(f'📋 Versión: {version}')
                tesseract_found = True
                break
            except Exception as e:
                print(f'⚠️  Error obteniendo versión: {e}')
    
    if not tesseract_found:
        print('❌ Tesseract no funciona correctamente')
        
except Exception as e:
    print(f'❌ Error con Tesseract: {e}')

# EasyOCR
try:
    import easyocr
    print('✅ EasyOCR disponible')
    
    # Probar crear reader
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print('✅ EasyOCR Reader creado exitosamente')
    
except Exception as e:
    print(f'❌ Error con EasyOCR: {e}')

print()
print('🚀 PRUEBA TU PROBLEMA ESPECIFICO:')
print('='*50)
print('Para probar \"B contiene AuB\":')
print('1. python run.py')
print('2. Cargar tu imagen')
print('3. Marcar ambos checkboxes (Tesseract + EasyOCR)')
print('4. Procesar con OCR')
print()
print('CONSEJOS PARA MEJOR RECONOCIMIENTO:')
print('• Usar imagen de alta resolución')
print('• Buen contraste (negro sobre blanco)')
print('• Símbolos grandes y claros')
print('• Si falla, probar escribir \"B contains A union B\"')
"

echo.
echo =========================================
echo   CONFIGURACION COMPLETADA
echo =========================================
echo.
echo ¿Quieres ejecutar la aplicación ahora? (S/N)
set /p choice="Selección: "

if /i "%choice%"=="S" (
    echo.
    echo Ejecutando aplicación...
    python run.py
) else (
    echo.
    echo Para ejecutar la aplicación más tarde:
    echo python run.py
    echo.
    pause
)
