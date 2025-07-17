"""
Configuración y utilidades adicionales para el escáner OCR
"""

import os
import sys
from pathlib import Path


class OCRConfig:
    """Configuración para los motores OCR"""
    
    # Configuración de Tesseract
    TESSERACT_CONFIGS = {
        'manuscrito': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz +-=()[]{}.,;:!?',
        'ecuaciones': r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+-=()[]{}.,;:!?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        'texto_normal': r'--oem 3 --psm 6',
        'numeros': r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+-=.,',
    }
    
    # Idiomas para EasyOCR
    EASYOCR_LANGUAGES = ['en', 'es']
    
    # Configuración de GPU para EasyOCR (cambiar a False si no tienes GPU)
    USE_GPU = False
    
    @staticmethod
    def get_tesseract_path():
        """Detecta automáticamente la ruta de Tesseract"""
        import shutil
        
        # Primero buscar en PATH
        tesseract_path = shutil.which('tesseract')
        if tesseract_path and os.path.exists(tesseract_path):
            return tesseract_path
        
        # Rutas comunes ordenadas por probabilidad
        possible_paths = [
            # Windows - rutas más comunes
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            r'C:\tesseract\tesseract.exe',
            r'C:\tools\tesseract\tesseract.exe',
            
            # Otras posibles ubicaciones en Windows
            r'D:\Program Files\Tesseract-OCR\tesseract.exe',
            r'D:\Tesseract-OCR\tesseract.exe',
            
            # Linux
            r'/usr/bin/tesseract',
            r'/usr/local/bin/tesseract',
            r'/opt/tesseract/bin/tesseract',
            
            # macOS
            r'/opt/homebrew/bin/tesseract',
            r'/usr/local/bin/tesseract',
            r'/opt/local/bin/tesseract'
        ]
        
        # Buscar en rutas comunes
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    @staticmethod
    def install_tesseract_instructions():
        """Devuelve instrucciones de instalación de Tesseract"""
        import platform
        system = platform.system().lower()
        
        if system == 'windows':
            return """
INSTRUCCIONES PARA INSTALAR TESSERACT EN WINDOWS:

1. Descargar Tesseract desde:
   https://github.com/UB-Mannheim/tesseract/wiki

2. Ejecutar el instalador como administrador

3. Durante la instalación:
   - Instalar en: C:\\Program Files\\Tesseract-OCR
   - Marcar "Add to PATH" si está disponible

4. Después de la instalación:
   - Reiniciar VS Code/terminal
   - Verificar con: tesseract --version

5. Si sigue sin funcionar, agregar manualmente al PATH:
   - Ir a Panel de Control > Sistema > Variables de entorno
   - Agregar C:\\Program Files\\Tesseract-OCR a la variable PATH

DESCARGA DIRECTA:
https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe
"""
        elif system == 'linux':
            return """
INSTRUCCIONES PARA INSTALAR TESSERACT EN LINUX:

Ubuntu/Debian:
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-spa  # Para español

CentOS/RHEL/Fedora:
sudo yum install tesseract
# o
sudo dnf install tesseract

Arch Linux:
sudo pacman -S tesseract
"""
        elif system == 'darwin':  # macOS
            return """
INSTRUCCIONES PARA INSTALAR TESSERACT EN macOS:

Con Homebrew (recomendado):
brew install tesseract

Con MacPorts:
sudo port install tesseract

Verificar instalación:
tesseract --version
"""
        else:
            return "Sistema no reconocido. Consulta la documentación oficial de Tesseract."
    
    @staticmethod
    def configure_tesseract():
        """Configura Tesseract automáticamente"""
        print("🔍 Buscando Tesseract...")
        
        # Primero intentar importar pytesseract
        try:
            import pytesseract
        except ImportError:
            print("❌ pytesseract no está instalado")
            print("Ejecuta: pip install pytesseract")
            return False
        
        # Buscar ruta de Tesseract
        tesseract_path = OCRConfig.get_tesseract_path()
        
        if tesseract_path:
            print(f"📍 Tesseract encontrado en: {tesseract_path}")
            
            try:
                # Configurar la ruta
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                print("⚙️  Configurando Tesseract...")
                
                # Probar que funciona con una imagen de prueba
                import numpy as np
                from PIL import Image
                
                # Crear imagen de prueba simple
                test_img = np.ones((100, 200), dtype=np.uint8) * 255
                test_img[30:70, 50:150] = 0  # Rectángulo negro (texto simulado)
                pil_img = Image.fromarray(test_img)
                
                # Intentar OCR de prueba
                try:
                    result = pytesseract.image_to_string(pil_img, config='--psm 6')
                    print("✅ Tesseract responde correctamente")
                    
                    # Obtener versión
                    version = pytesseract.get_tesseract_version()
                    print(f"📋 Versión de Tesseract: {version}")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Error en prueba de Tesseract: {e}")
                    
                    # Intentar configuración alternativa
                    if "tesseract is not installed" in str(e).lower():
                        print("🔧 Intentando configuración alternativa...")
                        
                        # Buscar en ubicaciones adicionales
                        alternative_paths = [
                            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                            tesseract_path.replace('tesseract', 'tesseract.exe') if not tesseract_path.endswith('.exe') else tesseract_path
                        ]
                        
                        for alt_path in alternative_paths:
                            if os.path.exists(alt_path):
                                print(f"� Probando ruta alternativa: {alt_path}")
                                pytesseract.pytesseract.tesseract_cmd = alt_path
                                try:
                                    test_result = pytesseract.image_to_string(pil_img, config='--psm 6')
                                    print("✅ Configuración alternativa exitosa")
                                    return True
                                except:
                                    continue
                        
                        print("❌ No se pudo configurar Tesseract automáticamente")
                        print("\n🛠️  CONFIGURACIÓN MANUAL:")
                        print("1. Abre Python y ejecuta:")
                        print("   import pytesseract")
                        print(f"   pytesseract.pytesseract.tesseract_cmd = r'{tesseract_path}'")
                        print("2. O edita el archivo main.py y agrega al inicio:")
                        print(f"   pytesseract.pytesseract.tesseract_cmd = r'{tesseract_path}'")
                        
                    return False
                    
            except Exception as e:
                print(f"❌ Error configurando Tesseract: {e}")
                return False
        else:
            print("❌ Tesseract no encontrado en rutas comunes")
            print("\n🔍 DIAGNÓSTICO:")
            
            # Verificar si está en PATH
            import subprocess
            try:
                result = subprocess.run(['tesseract', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✅ Tesseract está en PATH")
                    print(f"📋 Versión: {result.stdout.split()[1] if len(result.stdout.split()) > 1 else 'Desconocida'}")
                    
                    # Configurar usando comando del PATH
                    try:
                        import shutil
                        path_tesseract = shutil.which('tesseract')
                        if path_tesseract:
                            pytesseract.pytesseract.tesseract_cmd = path_tesseract
                            print(f"✅ Configurado usando PATH: {path_tesseract}")
                            return True
                    except:
                        pass
                else:
                    print("❌ Tesseract no responde correctamente")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print("❌ Tesseract no está en PATH")
            
            print(OCRConfig.install_tesseract_instructions())
            return False


class ImagePreprocessor:
    """Utilidades para preprocesamiento de imágenes"""
    
    @staticmethod
    def enhance_for_handwriting(image):
        """Optimiza imagen para reconocimiento de texto manuscrito"""
        import cv2
        import numpy as np
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aumentar el tamaño de la imagen para mejor OCR
        height, width = gray.shape
        if height < 100 or width < 200:
            scale_factor = max(200 / width, 100 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Aplicar filtro de mediana para reducir ruido
        denoised = cv2.medianBlur(gray, 3)
        
        # Mejorar contraste usando CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Umbralización adaptativa
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Operaciones morfológicas para mejorar la calidad
        kernel = np.ones((2,2), np.uint8)
        
        # Cerrar pequeños agujeros
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Abrir para eliminar ruido pequeño
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    @staticmethod
    def enhance_for_math_symbols(image):
        """Optimización mejorada para símbolos matemáticos manuscritos"""
        import cv2
        import numpy as np
        
        print("🔧 Aplicando preprocesamiento optimizado para símbolos matemáticos...")
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        print(f"   📐 Tamaño original: {gray.shape}")
        
        # Redimensionar moderadamente (no demasiado agresivo)
        height, width = gray.shape
        max_dim = max(height, width)
        
        if max_dim < 800:
            scale = 800 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            print(f"   📐 Redimensionado a: {resized.shape}")
        else:
            resized = gray.copy()
            print("   📐 Sin redimensionamiento necesario")
        
        # Reducir ruido con filtro bilateral (preserva bordes)
        denoised = cv2.bilateralFilter(resized, 9, 50, 50)
        
        # Mejora suave de contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Aplicar ligero sharpening para símbolos
        kernel_sharpen = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]]) * 0.15
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Binarización adaptativa suave
        adaptive_thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 8
        )
        
        # Operaciones morfológicas mínimas
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        
        # Cerrar pequeños espacios en símbolos
        closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_small)
        
        # Limpiar ruido pequeño sin afectar símbolos importantes
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
        
        print("   ✅ Preprocesamiento completado")
        
        return opened
        
        # Invertir si es necesario (texto negro sobre fondo blanco)
        if np.mean(thresh_otsu) < 127:
            thresh_otsu = cv2.bitwise_not(thresh_otsu)
        
        # Operaciones morfológicas refinadas
        # Kernel pequeño para preservar detalles de símbolos matemáticos
        kernel_small = np.ones((1,1), np.uint8)
        
        # Cerrar pequeños agujeros en los caracteres
        closed = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_small)
        
        # Abrir para limpiar ruido muy pequeño
        final = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small)
        
        return final
    
    @staticmethod
    def enhance_for_equations(image):
        """Optimiza imagen para reconocimiento de ecuaciones"""
        import cv2
        import numpy as np
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar desenfoque gaussiano suave
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Umbralización binaria simple
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas para limpiar
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    @staticmethod
    def deskew_image(image):
        """Corrige la inclinación de la imagen"""
        import cv2
        import numpy as np
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detectar bordes
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detectar líneas usando transformada de Hough
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Calcular ángulo promedio
            angles = []
            for rho, theta in lines[:10]:  # Usar solo las primeras 10 líneas
                angle = theta * 180 / np.pi
                angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                # Convertir a ángulo de rotación
                if avg_angle > 45:
                    rotation_angle = avg_angle - 90
                else:
                    rotation_angle = avg_angle
                
                # Rotar imagen
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
                return rotated
        
        return image
    
    @staticmethod
    def save_debug_image(image, filepath):
        """Guardar imagen para depuración"""
        import cv2
        try:
            cv2.imwrite(filepath, image)
            return True
        except Exception as e:
            print(f"⚠️  Error guardando imagen debug: {e}")
            return False


class LaTeXTemplates:
    """Plantillas y utilidades para LaTeX"""
    
    DOCUMENT_TEMPLATES = {
        'articulo': """\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[spanish]{{babel}}
\\usepackage{{amsmath, amsfonts, amssymb}}
\\usepackage{{geometry}}
\\geometry{{margin=2.5cm}}

\\title{{{title}}}
\\author{{{author}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

{content}

\\end{{document}}""",
        
        'notas': """\\documentclass[11pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[spanish]{{babel}}
\\usepackage{{amsmath, amsfonts, amssymb}}
\\usepackage{{geometry, fancyhdr}}
\\geometry{{margin=2cm}}

\\pagestyle{{fancy}}
\\fancyhf{{}}
\\fancyhead[L]{{{subject}}}
\\fancyhead[R]{{\\today}}
\\fancyfoot[C]{{\\thepage}}

\\begin{{document}}

{content}

\\end{{document}}""",
        
        'examen': """\\documentclass[12pt]{{exam}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[spanish]{{babel}}
\\usepackage{{amsmath, amsfonts, amssymb}}

\\begin{{document}}

\\begin{{questions}}
{content}
\\end{{questions}}

\\end{{document}}"""
    }
    
    @staticmethod
    def format_as_exercise(content, number=1):
        """Formatea contenido como ejercicio"""
        return f"""\\begin{{enumerate}}
\\setcounter{{enumi}}{{{number-1}}}
\\item {content}
\\end{{enumerate}}"""
    
    @staticmethod
    def format_as_equation_block(equations):
        """Formatea múltiples ecuaciones en un bloque"""
        if isinstance(equations, str):
            equations = [equations]
        
        formatted = "\\begin{align}\n"
        for i, eq in enumerate(equations):
            if i > 0:
                formatted += " \\\\\n"
            formatted += f"    {eq}"
        formatted += "\n\\end{align}"
        
        return formatted


def setup_environment():
    """Configura el entorno para la aplicación"""
    print("🔧 Configurando entorno...")
    
    # Configurar Tesseract
    tesseract_configured = OCRConfig.configure_tesseract()
    
    if not tesseract_configured:
        print("\n⚠️  TESSERACT NO ESTÁ DISPONIBLE")
        print("La aplicación funcionará solo con EasyOCR")
        print("Para usar Tesseract, sigue las instrucciones mostradas arriba")
    
    # Verificar disponibilidad de GPU para EasyOCR
    try:
        import torch
        if torch.cuda.is_available() and OCRConfig.USE_GPU:
            print("🚀 GPU disponible para EasyOCR")
        else:
            print("💻 Usando CPU para EasyOCR")
    except ImportError:
        print("⚠️  PyTorch no disponible")
    
    print("✅ Configuración completada")
    return tesseract_configured


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    dependencies = {
        'pytesseract': 'OCR con Tesseract',
        'easyocr': 'OCR avanzado',
        'cv2': 'Procesamiento de imágenes (opencv-python)',
        'PIL': 'Manejo de imágenes (Pillow)',
        'numpy': 'Cálculos numéricos',
        'PyQt5': 'Interfaz gráfica'
    }
    
    missing = []
    available = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            available.append(f"✅ {dep} - {description}")
        except ImportError:
            missing.append(f"❌ {dep} - {description}")
    
    print("\n📦 ESTADO DE DEPENDENCIAS:")
    print("-" * 40)
    
    for dep in available:
        print(dep)
    
    if missing:
        print("\nDEPENDENCIAS FALTANTES:")
        for dep in missing:
            print(dep)
        print("\nPara instalar las dependencias faltantes:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Todas las dependencias están instaladas")
        return True


if __name__ == "__main__":
    print("🧪 VERIFICANDO CONFIGURACIÓN DEL ESCÁNER OCR")
    print("=" * 50)
    
    # Verificar dependencias
    deps_ok = check_dependencies()
    
    if deps_ok:
        print("\n🔧 CONFIGURANDO ENTORNO...")
        setup_environment()
    else:
        print("\n❌ Faltan dependencias. Instálalas primero.")
    
    print("\n" + "=" * 50)
