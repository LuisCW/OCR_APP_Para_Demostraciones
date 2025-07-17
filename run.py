"""
Launcher mejorado para la aplicación Escáner OCR
"""

import sys
import os
from pathlib import Path

def check_environment():
    """Verifica el entorno antes de ejecutar"""
    print("🔍 Verificando entorno...")
    
    # Verificar dependencias críticas
    missing_deps = []
    
    try:
        import PyQt5
        print("✅ PyQt5 disponible")
    except ImportError:
        missing_deps.append("PyQt5")
    
    try:
        import cv2
        print("✅ OpenCV disponible")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
        print("✅ NumPy disponible")
    except ImportError:
        missing_deps.append("numpy")
    
    # Verificar motores OCR
    ocr_available = False
    
    try:
        import pytesseract
        print("✅ Tesseract disponible")
        ocr_available = True
    except ImportError:
        print("⚠️  Tesseract no disponible")
    
    try:
        import easyocr
        print("✅ EasyOCR disponible")
        ocr_available = True
    except ImportError:
        print("⚠️  EasyOCR no disponible")
    
    # Verificar motor de demostraciones
    try:
        import sympy
        print("✅ SymPy (demostraciones) disponible")
    except ImportError:
        missing_deps.append("sympy")
    
    if missing_deps:
        print(f"\n❌ Dependencias faltantes: {', '.join(missing_deps)}")
        print("Ejecuta: pip install -r requirements.txt")
        return False
    
    if not ocr_available:
        print("\n❌ No hay motores OCR disponibles")
        print("Instala al menos uno:")
        print("- pip install pytesseract")
        print("- pip install easyocr")
        print("- Para Tesseract: descargar desde tesseract-ocr.github.io")
        return False
    
    print("✅ Entorno verificado correctamente")
    return True

def main():
    """Función principal del launcher"""
    print("🚀 Escáner OCR con Motor de Demostraciones Matemáticas")
    print("=" * 60)
    
    # Verificar que estamos en el directorio correcto
    current_dir = Path(__file__).parent
    main_file = current_dir / "main.py"
    
    if not main_file.exists():
        print("❌ Error: No se encontró main.py")
        print(f"   Asegúrate de estar en el directorio correcto: {current_dir}")
        input("Presiona Enter para salir...")
        return
    
    # Verificar entorno
    if not check_environment():
        print("\n💡 SOLUCIONES:")
        print("1. Ejecutar el instalador: install_completo.bat")
        print("2. Instalar manualmente: pip install -r requirements.txt")
        print("3. Para Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        input("\nPresiona Enter para salir...")
        return
    
    print("\n🎯 Iniciando aplicación principal...")
    
    # Configurar entorno OCR
    try:
        from ocr_config import setup_environment
        setup_environment()
    except ImportError:
        print("⚠️  Módulo de configuración no encontrado")
    
    # Ejecutar aplicación principal
    try:
        import main
        main.main()
    except Exception as e:
        print(f"❌ Error al ejecutar la aplicación: {e}")
        print("\n💡 Posibles soluciones:")
        print("   1. Reinstalar dependencias: pip install -r requirements.txt")
        print("   2. Verificar instalación de Tesseract OCR")
        print("   3. Revisar que Python 3.7+ esté instalado")
        print("   4. Ejecutar como administrador si es necesario")
        input("\nPresiona Enter para salir...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Aplicación cerrada por el usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        input("Presiona Enter para salir...")
