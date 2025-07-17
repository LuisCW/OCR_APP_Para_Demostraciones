import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QFileDialog, QProgressBar, QTabWidget, QScrollArea,
                             QGridLayout, QComboBox, QCheckBox, QSpinBox,
                             QMessageBox, QSplitter, QFrame, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor
import cv2
import numpy as np
from PIL import Image
import re

# Importar dependencias OCR - SOLO EasyOCR para manuscritos matemáticos
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    # EasyOCR disponible y optimizado para manuscritos matemáticos
except ImportError:
    EASYOCR_AVAILABLE = False
    # EasyOCR no disponible - INSTALAR: pip install easyocr

# Solo EasyOCR - Tesseract eliminado completamente
TESSERACT_AVAILABLE = False

# Importar el motor de demostraciones
from proof_engine import ProofAssistant
from ocr_config import OCRConfig, setup_environment

# Importar optimizador matemático y preprocesador de teoría de conjuntos
try:
    from set_theory_ocr import SetTheoryOCRPreprocessor, MathSymbolCorrector
    SET_THEORY_AVAILABLE = True
    # Preprocesador de teoría de conjuntos disponible
except ImportError:
    SET_THEORY_AVAILABLE = False
    # Preprocesador de teoría de conjuntos no disponible

try:
    from math_ocr_optimizer import MathSymbolPreprocessor, MathOCROptimizer
    MATH_OPTIMIZER_AVAILABLE = True
    # Optimizador matemático disponible
except ImportError:
    MATH_OPTIMIZER_AVAILABLE = False
    # Optimizador matemático no disponible

# Importar sistema OCR mejorado
try:
    from enhanced_ocr_system import EnhancedOCREngine
    ENHANCED_OCR_AVAILABLE = True
    print("✅ Sistema OCR mejorado disponible")
except ImportError:
    ENHANCED_OCR_AVAILABLE = False
    print("⚠️ Sistema OCR mejorado no disponible - usando OCR estándar")


class OCRThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(str, str)  # placeholder_result, easyocr_result
    error = pyqtSignal(str)

    def __init__(self, image_path, use_easyocr=True):
        super().__init__()
        self.image_path = image_path
        self.use_easyocr = use_easyocr

    def run(self):
        try:
            placeholder_result = ""
            easyocr_result = ""
            
            # Verificar disponibilidad de EasyOCR
            if not EASYOCR_AVAILABLE:
                self.error.emit("EasyOCR no está instalado. Ejecuta: pip install easyocr")
                return
            
            self.progress.emit(10)
            
            # Cargar imagen
            image = cv2.imread(self.image_path)
            if image is None:
                self.error.emit(f"No se pudo cargar la imagen: {self.image_path}")
                return
            
            self.progress.emit(20)
            
            # Usar sistema OCR mejorado si está disponible
            if ENHANCED_OCR_AVAILABLE:
                print("� Usando sistema OCR mejorado con múltiples estrategias...")
                
                try:
                    # Inicializar motor OCR mejorado
                    enhanced_ocr = EnhancedOCREngine()
                    
                    self.progress.emit(30)
                    
                    # Realizar reconocimiento con múltiples estrategias
                    result = enhanced_ocr.recognize_with_multiple_strategies(image)
                    
                    self.progress.emit(80)
                    
                    if "error" in result:
                        easyocr_result = f"Error en OCR mejorado: {result['error']}"
                        print(f"❌ {easyocr_result}")
                    else:
                        easyocr_result = result["text"]
                        confidence = result.get("confidence", 0.0)
                        method = result.get("method", "desconocido")
                        math_score = result.get("math_score", 0.0)
                        
                        print(f"✅ OCR mejorado completado:")
                        print(f"   📝 Texto: '{easyocr_result}'")
                        print(f"   📊 Confianza: {confidence:.2f}")
                        print(f"   🧮 Puntaje matemático: {math_score:.2f}")
                        print(f"   🔧 Método: {method}")
                    
                except Exception as e:
                    print(f"⚠️ Error en OCR mejorado, usando fallback: {e}")
                    easyocr_result = self._fallback_ocr(image)
            
            else:
                print("📝 Usando OCR estándar (sistema mejorado no disponible)")
                easyocr_result = self._fallback_ocr(image)
            
            self.progress.emit(100)
            
            # Enviar resultados
            self.result.emit(placeholder_result, easyocr_result)
            
        except Exception as e:
            error_msg = f"Error general en OCR: {str(e)}"
            print(f"❌ {error_msg}")
            self.error.emit(error_msg)
    
    def _fallback_ocr(self, image):
        """OCR de respaldo usando el sistema original"""
        try:
            # Preprocesamiento especializado para teoría de conjuntos
            if SET_THEORY_AVAILABLE:
                print("🔧 Aplicando preprocesamiento para teoría de conjuntos...")
                set_theory_processor = SetTheoryOCRPreprocessor()
                processed_image = set_theory_processor.preprocess_for_set_theory(image)
            else:
                processed_image = self.preprocess_image(image)
            
            # EasyOCR con configuración básica
            if EASYOCR_AVAILABLE:
                print("🔍 Ejecutando OCR de respaldo con EasyOCR...")
                
                # Configurar EasyOCR
                reader = easyocr.Reader(['en'], gpu=False, verbose=False, download_enabled=False)
                
                # Configuración básica para evitar cuelgues
                config = {'detail': 1, 'paragraph': False}
                
                # Reconocimiento con timeout
                import time
                start_time = time.time()
                
                results = reader.readtext(processed_image, **config)
                
                elapsed_time = time.time() - start_time
                print(f"⏱️ OCR completado en {elapsed_time:.2f}s")
                
                # Procesar resultados
                if results:
                    texts = []
                    for (bbox, text, confidence) in results:
                        if confidence > 0.1:
                            texts.append(text)
                            print(f"   📝 Detectado: '{text}' (confianza: {confidence:.3f})")
                    
                    if texts:
                        result = ' '.join(texts)
                        # Aplicar correcciones básicas
                        result = self._correct_math_symbols(result)
                        return result
                
                return "No se pudo reconocer texto matemático"
            
            return "EasyOCR no disponible"
            
        except Exception as e:
            print(f"❌ Error en OCR de respaldo: {e}")
            return f"Error en OCR: {str(e)}"

    def preprocess_image(self, image):
        """Preprocesa la imagen para mejorar el OCR usando métodos optimizados para matemáticas manuscritas"""
        
        # Usar el optimizador matemático si está disponible
        if MATH_OPTIMIZER_AVAILABLE:
            try:
                print("🔬 Aplicando preprocesamiento especializado para símbolos matemáticos...")
                math_preprocessor = MathSymbolPreprocessor()
                processed = math_preprocessor.enhance_math_symbols(image)
                
                # Guardar imagen procesada para debug
                try:
                    debug_dir = os.path.dirname(self.image_path) if self.image_path else "."
                    cv2.imwrite(os.path.join(debug_dir, "debug_math_processed.png"), processed)
                    print("🖼️  Imagen debug guardada como: debug_math_processed.png")
                except Exception as debug_error:
                    print(f"⚠️  No se pudo guardar imagen debug: {debug_error}")
                
                return processed
                
            except Exception as e:
                print(f"⚠️  Error en preprocesamiento matemático: {e}")
                print("📝 Usando preprocesamiento estándar...")
        
        # Fallback al preprocesamiento estándar optimizado para manuscritos
        from ocr_config import ImagePreprocessor
        
        # Guardar imagen original para debug
        try:
            debug_dir = os.path.dirname(self.image_path) if self.image_path else "."
            cv2.imwrite(os.path.join(debug_dir, "debug_original.png"), image)
        except Exception as debug_error:
            print(f"⚠️  No se pudo guardar imagen original debug: {debug_error}")
        
        # Aplicar preprocesamiento optimizado para manuscritos matemáticos
        processed = ImagePreprocessor.enhance_for_math_symbols(image)
        
        # Guardar imagen procesada para debug
        try:
            cv2.imwrite(os.path.join(debug_dir, "debug_processed.png"), processed)
            print("🖼️  Imágenes debug guardadas en el directorio de la imagen")
        except Exception as debug_error:
            print(f"⚠️  No se pudo guardar imagen procesada debug: {debug_error}")
        
        return processed

    def _evaluate_math_content(self, text):
        """Evalúa la calidad y contenido matemático del texto reconocido"""
        if not text or not text.strip():
            return 0.0
        
        score = 0.0
        text_clean = text.strip().lower()
        
        # Puntos por símbolos matemáticos
        math_symbols = ['=', '+', '-', '*', '/', '^', '_', '∪', '∩', '⊆', '⊇', '⊃', '⊂', '∈', '∉']
        for symbol in math_symbols:
            if symbol in text:
                score += 1.0
        
        # Puntos por letras (variables matemáticas)
        if any(c.isalpha() for c in text):
            score += 0.5
        
        # Puntos por números
        if any(c.isdigit() for c in text):
            score += 0.3
        
        # Penalización por caracteres extraños
        strange_chars = ['|', '~', '`', '#', '$', '%']
        for char in strange_chars:
            if char in text:
                score -= 0.5
        
        # Bonus por longitud razonable
        if 2 <= len(text.strip()) <= 50:
            score += 0.5
        
        return max(0.0, score)

    def _correct_math_symbols(self, text):
        """Corregir símbolos matemáticos comunes mal reconocidos"""
        if not text:
            return text
        
        print(f"🔧 Corrigiendo texto: '{text}'")
        
        # Paso 1: Correcciones específicas para casos problemáticos comunes
        specific_corrections = {
            # Casos específicos observados
            'B c AuB': 'B ⊇ A ∪ B',
            'B c A u B': 'B ⊇ A ∪ B',
            'B c A U B': 'B ⊇ A ∪ B',
            'B C AuB': 'B ⊇ A ∪ B',
            'B C A u B': 'B ⊇ A ∪ B',
            'B C A U B': 'B ⊇ A ∪ B',
            'A c B': 'A ⊆ B',
            'A C B': 'A ⊇ B',
            
            # Patrones con espacios
            ' c ': ' ⊇ ',  # 'contains' mal reconocido
            ' C ': ' ⊇ ',  # 'Contains' mal reconocido
        }
        
        corrected = text
        
        # Aplicar correcciones específicas primero
        for wrong, right in specific_corrections.items():
            corrected = corrected.replace(wrong, right)
        
        # Paso 2: Correcciones generales de símbolos
        general_corrections = {
            # Símbolos de unión - más agresivo
            ' U ': ' ∪ ',
            'U': '∪',  # U sola
            ' u ': ' ∪ ',
            'union': '∪',
            'Au': 'A ∪',
            'Bu': 'B ∪',
            'AuB': 'A ∪ B',
            'BuA': 'B ∪ A',
            'AUB': 'A ∪ B',
            'BUA': 'B ∪ A',
            
            # Símbolos de intersección
            ' n ': ' ∩ ',
            ' ^ ': ' ∩ ',
            'intersection': '∩',
            'AnB': 'A ∩ B',
            'BnA': 'B ∩ A',
            
            # Símbolos de contención - más agresivo
            'contains': '⊇',
            'subset': '⊆',
            'superset': '⊇',
            
            # Pertenencia
            ' E ': ' ∈ ',
            ' e ': ' ∈ ',
            'element': '∈',
            'belongs': '∈',
            'in': '∈',
            
            # Conjunto vacío
            ' O ': ' ∅ ',
            ' 0 ': ' ∅ ',
            'empty': '∅',
            'null': '∅',
            
            # Limpiar caracteres problemáticos
            '|': '',
            '_': '',
            '~': '',
            '#': '',
            '$': '',
            '%': '',
            '@': ''
        }
        
        # Aplicar correcciones generales
        for wrong, right in general_corrections.items():
            corrected = corrected.replace(wrong, right)
        
        # Paso 3: Correcciones basadas en patrones regex para casos complejos
        import re
        
        # Patrón: letra + c/C + letra (probablemente contención)
        corrected = re.sub(r'([A-Z])\s*[cC]\s*([A-Z])', r'\1 ⊇ \2', corrected)
        
        # Patrón: letra + u/U + letra (probablemente unión)
        corrected = re.sub(r'([A-Z])\s*[uU]\s*([A-Z])', r'\1 ∪ \2', corrected)
        
        # Patrón: letra + n + letra (probablemente intersección)
        corrected = re.sub(r'([A-Z])\s*n\s*([A-Z])', r'\1 ∩ \2', corrected)
        
        # Limpiar espacios extra
        corrected = ' '.join(corrected.split())
        
        print(f"✅ Texto corregido: '{corrected}'")
        return corrected


class LaTeXConverter:
    def __init__(self):
        self.math_patterns = {
            # Fracciones
            r'(\d+)\s*/\s*(\d+)': r'\\frac{\1}{\2}',
            # Exponentes
            r'(\w+)\^(\w+)': r'\1^{\2}',
            r'(\w+)\^(\d+)': r'\1^{\2}',
            # Subíndices
            r'(\w+)_(\w+)': r'\1_{\2}',
            r'(\w+)_(\d+)': r'\1_{\2}',
            # Raíces cuadradas
            r'sqrt\s*\(([^)]+)\)': r'\\sqrt{\1}',
            r'√\s*\(([^)]+)\)': r'\\sqrt{\1}',
            # Integrales
            r'∫': r'\\int',
            r'integral': r'\\int',
            # Sumatorias
            r'∑': r'\\sum',
            r'suma': r'\\sum',
            # Símbolos griegos
            r'\balpha\b': r'\\alpha',
            r'\bbeta\b': r'\\beta',
            r'\bgamma\b': r'\\gamma',
            r'\bdelta\b': r'\\delta',
            r'\btheta\b': r'\\theta',
            r'\blambda\b': r'\\lambda',
            r'\bmu\b': r'\\mu',
            r'\bpi\b': r'\\pi',
            r'\bsigma\b': r'\\sigma',
            r'\btau\b': r'\\tau',
            r'\bphi\b': r'\\phi',
            r'\bomega\b': r'\\omega',
            # Operadores
            r'<=': r'\\leq',
            r'>=': r'\\geq',
            r'!=': r'\\neq',
            r'±': r'\\pm',
            r'∞': r'\\infty',
        }

    def convert_to_latex(self, text):
        """Convierte texto a formato LaTeX"""
        if not text.strip():
            return ""
        
        latex_text = text.strip()
        
        # Aplicar patrones matemáticos
        for pattern, replacement in self.math_patterns.items():
            latex_text = re.sub(pattern, replacement, latex_text, flags=re.IGNORECASE)
        
        # Detectar y formatear ecuaciones
        latex_text = self.format_equations(latex_text)
        
        # Generar documento LaTeX completo
        latex_document = self.generate_latex_document(latex_text)
        
        return latex_document

    def format_equations(self, text):
        """Detecta y formatea ecuaciones matemáticas"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
                
            # Detectar si la línea contiene matemáticas
            if self.is_math_line(line):
                # Formatear como ecuación
                formatted_lines.append(f'\\begin{{equation}}')
                formatted_lines.append(f'{line}')
                formatted_lines.append(f'\\end{{equation}}')
            else:
                # Texto normal
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def is_math_line(self, line):
        """Determina si una línea contiene contenido matemático"""
        math_indicators = [
            '=', '+', '-', '*', '/', '^', '_', '\\frac', '\\sqrt',
            '\\int', '\\sum', '\\alpha', '\\beta', '\\gamma', '\\delta',
            '\\theta', '\\lambda', '\\mu', '\\pi', '\\sigma', '\\phi',
            '\\omega', '\\leq', '\\geq', '\\neq', '\\pm', '\\infty'
        ]
        
        return any(indicator in line for indicator in math_indicators)

    def generate_latex_document(self, content):
        """Genera un documento LaTeX completo"""
        return f"""\\documentclass{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[spanish]{{babel}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{geometry}}
\\geometry{{margin=2cm}}

\\title{{Documento Escaneado}}
\\author{{Generado por OCR Scanner}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

{content}

\\end{{document}}"""


class ProofThread(QThread):
    """Hilo para generar demostraciones matemáticas"""
    progress = pyqtSignal(int)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, text, proof_method='auto'):
        super().__init__()
        self.text = text
        self.proof_method = proof_method
        self.proof_assistant = ProofAssistant()

    def run(self):
        try:
            self.progress.emit(20)
            
            # Analizar el problema
            analysis = self.proof_assistant.analyze_problem(self.text)
            self.progress.emit(50)
            
            # NO FORZAR TIPO - Permitir que la detección automática funcione
            # Si queremos forzar un método específico, mantenemos el tipo detectado
            if self.proof_method == 'gentzen':
                # Solo forzar si no se detectó un tipo específico
                if analysis['type'] in ['direct_proof', 'unknown']:
                    analysis['type'] = 'gentzen_logic'
            elif self.proof_method == 'induction':
                # Solo forzar si no se detectó un tipo específico
                if analysis['type'] in ['direct_proof', 'unknown']:
                    analysis['type'] = 'induction'
            
            proof = self.proof_assistant.generate_proof(analysis)
            self.progress.emit(90)
            
            # Combinar análisis y demostración
            result = {
                'analysis': analysis,
                'proof': proof
            }
            
            self.progress.emit(100)
            self.result.emit(result)
            
        except Exception as e:
            self.error.emit(f"Error en generación de demostración: {str(e)}")


class ScannerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.latex_converter = LaTeXConverter()
        self.proof_assistant = ProofAssistant()
        self.current_image_path = None
        self.ocr_thread = None
        self.proof_thread = None
        
        self.setWindowTitle("Escáner OCR con Motor de Demostraciones Matemáticas")
        self.setGeometry(100, 100, 1600, 1000)
        self.setup_ui()
        self.setup_style()

    def setup_ui(self):
        """Configura la interfaz de usuario"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Panel izquierdo - Controles
        left_panel = self.create_left_panel()
        
        # Panel derecho - Resultados
        right_panel = self.create_right_panel()
        
        # Splitter para dividir paneles
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)

    def create_left_panel(self):
        """Crea el panel izquierdo con controles"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Título
        title = QLabel("Escáner de Demostraciones")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Botón para cargar imagen
        self.load_button = QPushButton("📁 Cargar Imagen")
        self.load_button.setMinimumHeight(40)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)
        
        # Vista previa de imagen
        self.image_label = QLabel("No hay imagen cargada")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9;")
        layout.addWidget(self.image_label)
        
        # Configuraciones OCR
        ocr_group = QGroupBox("Configuración de Reconocimiento")
        ocr_layout = QVBoxLayout(ocr_group)
        
        # EasyOCR checkbox con estado basado en disponibilidad
        self.easyocr_checkbox = QCheckBox("✅ Reconocimiento OCR Activado")
        if EASYOCR_AVAILABLE:
            self.easyocr_checkbox.setChecked(True)
            self.easyocr_checkbox.setToolTip("OCR optimizado para manuscritos matemáticos y teoría de conjuntos")
        else:
            self.easyocr_checkbox.setChecked(False)
            self.easyocr_checkbox.setEnabled(False)
            self.easyocr_checkbox.setToolTip("EasyOCR no está instalado")
            self.easyocr_checkbox.setText("❌ OCR No Disponible")
        ocr_layout.addWidget(self.easyocr_checkbox)
        
        if not EASYOCR_AVAILABLE:
            info_label = QLabel()
            info_text = "⚠️ El reconocimiento OCR no está disponible.\nEjecuta: pip install easyocr"
            
            info_label.setText(info_text)
            info_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; background-color: #ffebee; border-radius: 4px;")
            info_label.setWordWrap(True)
            ocr_layout.addWidget(info_label)
        else:
            # Mensaje informativo sobre optimización
            info_label = QLabel()
            info_text = "✅ Reescribe tus formulas, utilizando notación matemática clara. Utiliza Gentzen e Induccion Matematica"
            
            info_label.setText(info_text)
            info_label.setStyleSheet("color: #2e7d32; font-size: 10px; padding: 5px; background-color: #e8f5e8; border-radius: 4px;")
            info_label.setWordWrap(True)
            ocr_layout.addWidget(info_label)
        
        layout.addWidget(ocr_group)
        
        # Configuraciones de demostración
        proof_group = QGroupBox("Motor de Demostraciones")
        proof_layout = QVBoxLayout(proof_group)
        
        proof_layout.addWidget(QLabel("Método de demostración:"))
        self.proof_method_combo = QComboBox()
        self.proof_method_combo.addItems([
            "Automático",
            "Gentzen (Cálculo de Secuentes)",
            "Inducción Matemática",
        ])
        proof_layout.addWidget(self.proof_method_combo)
        

        
        layout.addWidget(proof_group)
        
        # Botones de procesamiento
        self.process_button = QPushButton("�️ Procesar Imagen")
        self.process_button.setMinimumHeight(40)
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_image)
        layout.addWidget(self.process_button)
        
        self.proof_button = QPushButton("🧮 Generar Demostración")
        self.proof_button.setMinimumHeight(40)
        self.proof_button.setEnabled(False)
        self.proof_button.clicked.connect(self.generate_proof)
        layout.addWidget(self.proof_button)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Botón de conversión a LaTeX
        self.latex_button = QPushButton("📄 Generar LaTeX")
        self.latex_button.setMinimumHeight(40)
        self.latex_button.setEnabled(False)
        self.latex_button.clicked.connect(self.generate_latex)
        layout.addWidget(self.latex_button)
        
        # Botón para guardar
        self.save_button = QPushButton("💾 Guardar Resultados")
        self.save_button.setMinimumHeight(40)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_results)
        layout.addWidget(self.save_button)
        
        layout.addStretch()
        return panel

    def create_right_panel(self):
        """Crea el panel derecho con resultados"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Título
        title = QLabel("Resultados y Demostraciones")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Pestañas para diferentes resultados
        self.tab_widget = QTabWidget()
        
        # Pestaña EasyOCR (único motor)
        self.easyocr_text = QTextEdit()
        self.easyocr_text.setFont(QFont("Courier", 10))
        self.easyocr_text.setPlaceholderText("El texto reconocido aparecerá aquí después del procesamiento...")
        self.tab_widget.addTab(self.easyocr_text, "� Texto Reconocido")
        
        # Pestaña Análisis del Problema
        self.analysis_text = QTextEdit()
        self.analysis_text.setFont(QFont("Courier", 10))
        self.analysis_text.setPlaceholderText("Análisis automático del problema aparecerá aquí...")
        self.tab_widget.addTab(self.analysis_text, "Análisis")
        
        # Pestaña Demostración
        self.proof_text = QTextEdit()
        self.proof_text.setFont(QFont("Courier", 10))
        self.proof_text.setPlaceholderText("Demostración matemática aparecerá aquí...")
        self.tab_widget.addTab(self.proof_text, "Demostración")
        
        # Pestaña LaTeX
        self.latex_text = QTextEdit()
        self.latex_text.setFont(QFont("Courier", 10))
        self.latex_text.setPlaceholderText("Código LaTeX generado aparecerá aquí...")
        self.tab_widget.addTab(self.latex_text, "Código LaTeX")
        
        layout.addWidget(self.tab_widget)
        return panel

    def setup_style(self):
        """Configura el estilo de la aplicación"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
            QFrame {
                background-color: white;
                border-radius: 8px;
            }
        """)

    def load_image(self):
        """Carga una imagen desde archivo"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Mostrar vista previa
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            
            # Habilitar botón de procesamiento
            self.process_button.setEnabled(True)
            
            # Limpiar resultados anteriores (solo EasyOCR)
            self.easyocr_text.clear()
            self.analysis_text.clear()
            self.proof_text.clear()
            self.latex_text.clear()
            self.latex_button.setEnabled(False)
            self.proof_button.setEnabled(False)
            self.save_button.setEnabled(False)

    def process_image(self):
        """Procesa la imagen con EasyOCR optimizado para manuscritos matemáticos"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Error", "No hay imagen seleccionada")
            return
        
        if not self.easyocr_checkbox.isChecked():
            QMessageBox.warning(self, "Advertencia", "El reconocimiento OCR debe estar habilitado para procesar la imagen")
            return
        
        if not EASYOCR_AVAILABLE:
            QMessageBox.critical(self, "Error", "El reconocimiento OCR no está disponible.\nEjecuta: pip install easyocr")
            return
        
        # Mostrar barra de progreso
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_button.setEnabled(False)
        
        print("🚀 Iniciando procesamiento optimizado para manuscritos matemáticos...")
        
        # Iniciar procesamiento en hilo separado (solo EasyOCR)
        self.ocr_thread = OCRThread(
            self.current_image_path,
            self.easyocr_checkbox.isChecked()
        )
        self.ocr_thread.progress.connect(self.progress_bar.setValue)
        self.ocr_thread.result.connect(self.on_ocr_result)
        self.ocr_thread.error.connect(self.on_ocr_error)
        self.ocr_thread.start()

    def on_ocr_result(self, placeholder_result, easyocr_result):
        """Maneja el resultado del OCR"""
        # Solo mostrar resultado de EasyOCR
        self.easyocr_text.setText(easyocr_result)
        
        # Ocultar barra de progreso
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        self.latex_button.setEnabled(True)
        self.proof_button.setEnabled(True)
        
        print(f"✅ OCR completado: {len(easyocr_result)} caracteres reconocidos")
        
        # Generación automática de demostraciones deshabilitada
        # if hasattr(self, 'auto_proof_checkbox') and self.auto_proof_checkbox.isChecked():
        #     if easyocr_result.strip():
        #         print("🧮 Generando demostración automáticamente...")
        #         self.generate_proof_for_text(easyocr_result)
        
        QMessageBox.information(self, "Éxito", "Procesamiento OCR completado")

    def on_ocr_error(self, error_message):
        """Maneja errores del OCR"""
        self.progress_bar.setVisible(False)
        self.process_button.setEnabled(True)
        
        # Mostrar error en la pestaña de texto reconocido
        error_text = f"Error en el procesamiento OCR:\n{error_message}"
        self.easyocr_text.setText(error_text)
        
        QMessageBox.critical(self, "Error", f"Error en el procesamiento OCR:\n{error_message}")

    def generate_proof_from_text(self, text: str):
        """Genera demostración automáticamente desde texto"""
        if not text.strip():
            return
        
        # Mostrar barra de progreso
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Determinar método de demostración
        method_map = {
            0: 'auto',          # Automático
            1: 'gentzen',       # Gentzen
            2: 'induction',     # Inducción
            3: 'direct'         # Directa
        }
        
        proof_method = method_map.get(self.proof_method_combo.currentIndex(), 'auto')
        
        # Iniciar generación de demostración
        self.proof_thread = ProofThread(text, proof_method)
        self.proof_thread.progress.connect(self.progress_bar.setValue)
        self.proof_thread.result.connect(self.on_proof_result)
        self.proof_thread.error.connect(self.on_proof_error)
        self.proof_thread.start()

    def generate_proof(self):
        """Genera demostración desde el texto reconocido por EasyOCR"""
        # Obtener texto de EasyOCR
        source_text = self.easyocr_text.toPlainText()
        
        if not source_text.strip():
            QMessageBox.warning(self, "Advertencia", 
                              "No hay texto reconocido para procesar.\nPrimero ejecuta el reconocimiento OCR.")
            return
        
        print(f"🧮 Generando demostración para: '{source_text.strip()}'")
        self.generate_proof_from_text(source_text)
    
    def generate_proof_for_text(self, text):
        """Genera demostración automáticamente para un texto dado"""
        if text and text.strip():
            self.generate_proof_from_text(text)

    def on_proof_result(self, result):
        """Maneja el resultado de la generación de demostración"""
        analysis = result['analysis']
        proof = result['proof']
        
        # Mostrar análisis (con verificación de componentes)
        components = analysis.get('components', {})
        premises = components.get('premises', [])
        variables = components.get('variables', [])
        quantifiers = components.get('quantifiers', [])
        operators = components.get('operators', [])
        
        analysis_text = f"""ANÁLISIS DEL PROBLEMA:

Tipo detectado: {analysis['type']}
Confianza: {analysis['confidence']:.2%}

Componentes encontrados:
- Premisas: {len(premises)}
- Variables: {', '.join(variables) if variables else 'Ninguna'}
- Cuantificadores: {', '.join(quantifiers) if quantifiers else 'Ninguno'}
- Operadores: {', '.join(operators) if operators else 'Ninguno'}

Texto original:
{analysis['text']}
"""
        self.analysis_text.setText(analysis_text)
        
        # Mostrar demostración
        if proof['success']:
            # Si hay texto de demostración específico (como Gentzen), usarlo
            if 'proof_text' in proof:
                proof_display = proof['proof_text']
            else:
                proof_display = f"""DEMOSTRACIÓN GENERADA:

Método utilizado: {proof['method']}

{proof.get('explanation', 'Demostración completada exitosamente.')}

LaTeX generado disponible en la pestaña correspondiente.
"""
            
            self.proof_text.setText(proof_display)
            
            # Actualizar LaTeX con la demostración
            if 'latex' in proof:
                current_latex = self.latex_text.toPlainText()
                if current_latex.strip():
                    # Combinar LaTeX existente con demostración
                    combined_latex = current_latex + "\n\n" + proof['latex']
                else:
                    combined_latex = proof['latex']
                self.latex_text.setText(combined_latex)
                
            self.save_button.setEnabled(True)
            
        else:
            error_text = f"""ERROR EN LA DEMOSTRACIÓN:

Método intentado: {proof['method']}
Error: {proof.get('error', 'Error desconocido')}

Sugerencias:
- Verificar que el texto esté completo y bien formateado
- Intentar con un método de demostración diferente
- Revisar la sintaxis matemática
"""
            self.proof_text.setText(error_text)
        
        # Cambiar a pestaña de análisis
        self.tab_widget.setCurrentIndex(2)
        
        # Ocultar barra de progreso
        self.progress_bar.setVisible(False)

    def on_proof_error(self, error_message):
        """Maneja errores en la generación de demostración"""
        self.progress_bar.setVisible(False)
        self.proof_text.setText(f"Error en la generación de demostración:\n{error_message}")
        QMessageBox.critical(self, "Error", f"Error en demostración:\n{error_message}")

    def generate_latex(self):
        """Genera código LaTeX desde el texto OCR"""
        # Usar el texto de la pestaña activa
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Texto Reconocido (EasyOCR)
            source_text = self.easyocr_text.toPlainText()
        elif current_tab == 1:  # Análisis
            QMessageBox.information(self, "Información", "Selecciona la pestaña 'Texto Reconocido' para generar LaTeX")
            return
        elif current_tab == 2:  # Demostración
            QMessageBox.information(self, "Información", "Selecciona la pestaña 'Texto Reconocido' para generar LaTeX")
            return
        elif current_tab == 3:  # Código LaTeX
            QMessageBox.information(self, "Información", "Selecciona la pestaña 'Texto Reconocido' para generar LaTeX")
            return
        else:
            QMessageBox.warning(self, "Advertencia", "Selecciona la pestaña 'Texto Reconocido' para generar LaTeX")
            return
        
        if not source_text.strip():
            QMessageBox.warning(self, "Advertencia", "No hay texto para convertir a LaTeX")
            return
        
        # Convertir a LaTeX
        latex_code = self.latex_converter.convert_to_latex(source_text)
        self.latex_text.setText(latex_code)
        
        # Cambiar a pestaña LaTeX (índice 4)
        self.tab_widget.setCurrentIndex(4)
        self.save_button.setEnabled(True)
        
        QMessageBox.information(self, "Éxito", "Código LaTeX generado correctamente")

    def save_results(self):
        """Guarda todos los resultados en archivos"""
        if not any([
            self.easyocr_text.toPlainText().strip(),
            self.latex_text.toPlainText().strip(),
            self.proof_text.toPlainText().strip()
        ]):
            QMessageBox.warning(self, "Advertencia", "No hay resultados para guardar")
            return
        
        # Seleccionar directorio
        directory = QFileDialog.getExistingDirectory(
            self, "Seleccionar directorio para guardar resultados"
        )
        
        if not directory:
            return
        
        try:
            base_name = "resultados_escaner"
            
            # Guardar OCR - EasyOCR (Texto Reconocido)
            if self.easyocr_text.toPlainText().strip():
                with open(f"{directory}/{base_name}_texto_reconocido.txt", 'w', encoding='utf-8') as f:
                    f.write(self.easyocr_text.toPlainText())
            
            # Guardar análisis
            if self.analysis_text.toPlainText().strip():
                with open(f"{directory}/{base_name}_analisis.txt", 'w', encoding='utf-8') as f:
                    f.write(self.analysis_text.toPlainText())
            
            # Guardar demostración
            if self.proof_text.toPlainText().strip():
                with open(f"{directory}/{base_name}_demostracion.txt", 'w', encoding='utf-8') as f:
                    f.write(self.proof_text.toPlainText())
            
            # Guardar LaTeX
            if self.latex_text.toPlainText().strip():
                with open(f"{directory}/{base_name}_completo.tex", 'w', encoding='utf-8') as f:
                    f.write(self.latex_text.toPlainText())
            
            QMessageBox.information(self, "Éxito", 
                                  f"Resultados guardados exitosamente en:\n{directory}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al guardar archivos:\n{str(e)}")

    def save_latex(self):
        """Guarda el código LaTeX en un archivo"""
        latex_content = self.latex_text.toPlainText()
        
        if not latex_content.strip():
            QMessageBox.warning(self, "Advertencia", "No hay código LaTeX para guardar")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Archivo LaTeX", "demostracion.tex",
            "Archivos LaTeX (*.tex);;Todos los archivos (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                QMessageBox.information(self, "Éxito", f"Archivo LaTeX guardado en:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al guardar archivo:\n{str(e)}")

    # ...existing code...


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Escáner OCR con Motor de Demostraciones Matemáticas")
    
    # Solo verificar que EasyOCR esté disponible
    if not EASYOCR_AVAILABLE:
        QMessageBox.critical(None, "Error Crítico", 
                           "EasyOCR no está disponible.\n\n"
                           "SOLUCIÓN:\n"
                           "Ejecuta: pip install easyocr\n"
                           "Luego reinicia la aplicación")
        sys.exit(1)
    
    # Configurar entorno (sin mensaje emergente)
    setup_environment()
    
    # Crear y mostrar ventana principal directamente
    window = ScannerGUI()
    window.show()
    
    # print("✅ Aplicación iniciada correctamente")  # Silencioso
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
