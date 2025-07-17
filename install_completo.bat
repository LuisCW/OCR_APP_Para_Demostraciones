@echo off
echo ========================================
echo  INSTALADOR COMPLETO - ESCANER OCR
echo  con Motor de Demostraciones Matematicas
echo ========================================
echo.

REM Verificar si se esta ejecutando como administrador
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ATENCION: Se recomienda ejecutar como administrador
    echo para instalar Tesseract automaticamente.
    echo.
    pause
)

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

echo Instalando dependencias Python...
echo.
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Verificando Tesseract OCR...

REM Verificar si Tesseract ya esta instalado
tesseract --version >nul 2>&1
if %errorLevel% equ 0 (
    echo Tesseract ya esta instalado.
    tesseract --version
    goto :CONFIGURE
)

REM Buscar Tesseract en rutas comunes
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo Tesseract encontrado en C:\Program Files\Tesseract-OCR\
    set "TESSERACT_PATH=C:\Program Files\Tesseract-OCR"
    goto :ADD_TO_PATH
)

if exist "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" (
    echo Tesseract encontrado en C:\Program Files (x86)\Tesseract-OCR\
    set "TESSERACT_PATH=C:\Program Files (x86)\Tesseract-OCR"
    goto :ADD_TO_PATH
)

echo Tesseract no encontrado. Descargando e instalando...
echo.

REM Crear directorio temporal
if not exist "%TEMP%\escaner_ocr" mkdir "%TEMP%\escaner_ocr"
cd /d "%TEMP%\escaner_ocr"

echo Descargando Tesseract OCR...
echo URL: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe

REM Intentar descargar con PowerShell
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe' -OutFile 'tesseract-installer.exe'}"

if exist "tesseract-installer.exe" (
    echo Descarga completada. Ejecutando instalador...
    echo.
    echo IMPORTANTE: Durante la instalacion:
    echo 1. Instalar en la ruta por defecto
    echo 2. Marcar "Add to PATH" si esta disponible
    echo 3. Instalar paquetes de idioma Espanol
    echo.
    start /wait tesseract-installer.exe
    
    REM Limpiar archivo temporal
    del tesseract-installer.exe
    cd /d "%~dp0"
    rmdir /s /q "%TEMP%\escaner_ocr"
) else (
    echo ERROR: No se pudo descargar Tesseract.
    echo Por favor descargalo manualmente desde:
    echo https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    goto :MANUAL_INSTALL
)

goto :CONFIGURE

:ADD_TO_PATH
echo Agregando Tesseract al PATH...
setx PATH "%PATH%;%TESSERACT_PATH%" /M >nul 2>&1
if %errorLevel% neq 0 (
    echo No se pudo agregar al PATH automaticamente.
    echo Agregalo manualmente: %TESSERACT_PATH%
)
goto :CONFIGURE

:MANUAL_INSTALL
echo ========================================
echo    INSTALACION MANUAL DE TESSERACT
echo ========================================
echo.
echo 1. Ve a: https://github.com/UB-Mannheim/tesseract/wiki
echo 2. Descarga la version para Windows
echo 3. Ejecuta el instalador como administrador
echo 4. Asegurate de marcar "Add to PATH"
echo 5. Reinicia esta aplicacion
echo.
pause
exit /b 1

:CONFIGURE
echo.
echo Configurando entorno...
python ocr_config.py

echo.
echo ========================================
echo       INSTALACION COMPLETADA!
echo ========================================
echo.

echo Verificando instalacion...
python -c "from ocr_config import check_dependencies; check_dependencies()"

echo.
echo Para ejecutar la aplicacion:
echo python main.py
echo.
echo O ejecuta directamente:
echo python run.py
echo.

pause
