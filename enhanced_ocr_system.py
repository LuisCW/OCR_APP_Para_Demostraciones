"""
Sistema OCR Mejorado para Fórmulas Matemáticas Manuscritas
Combina múltiples técnicas de preprocesamiento y reconocimiento
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import re
from typing import List, Tuple, Dict, Optional
import time

# Importar analizador espacial
try:
    from spatial_math_analyzer import SpatialMathAnalyzer
    SPATIAL_ANALYZER_AVAILABLE = True
    print("✅ Analizador espacial disponible")
except ImportError:
    SPATIAL_ANALYZER_AVAILABLE = False
    print("⚠️ Analizador espacial no disponible")

class AdvancedImagePreprocessor:
    """Preprocesador avanzado con múltiples estrategias para diferentes tipos de escritura"""
    
    def __init__(self):
        self.debug_mode = True
        
    def create_multiple_variants(self, image: np.ndarray) -> List[np.ndarray]:
        """Crea múltiples variantes optimizadas de la imagen"""
        variants = []
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Variante 1: Optimización para escritura clara
        variant1 = self._optimize_for_clear_writing(gray)
        variants.append(("clear_writing", variant1))
        
        # Variante 2: Optimización para escritura borrosa
        variant2 = self._optimize_for_blurry_writing(gray)
        variants.append(("blurry_writing", variant2))
        
        # Variante 3: Optimización para símbolos pequeños
        variant3 = self._optimize_for_small_symbols(gray)
        variants.append(("small_symbols", variant3))
        
        # Variante 4: Máximo contraste para escritura débil
        variant4 = self._optimize_for_weak_writing(gray)
        variants.append(("weak_writing", variant4))
        
        # Variante 5: Denoising agresivo para escritura con ruido
        variant5 = self._optimize_for_noisy_writing(gray)
        variants.append(("noisy_writing", variant5))
        
        # Variante 6: Optimización para estructuras matemáticas complejas (NUEVA)
        variant6 = self._optimize_for_math_structures(gray)
        variants.append(("math_structures", variant6))
        
        return variants
    
    def _optimize_for_clear_writing(self, image: np.ndarray) -> np.ndarray:
        """Optimización para escritura manuscrita clara"""
        # Redimensionar conservando calidad
        height, width = image.shape
        scale = max(2.0, 800 / max(height, width))
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Filtro bilateral para suavizar preservando bordes
        filtered = cv2.bilateralFilter(resized, 9, 75, 75)
        
        # Mejora de contraste adaptativo
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Binarización adaptativa Gaussian
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Operación morfológica suave para conectar líneas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return result
    
    def _optimize_for_blurry_writing(self, image: np.ndarray) -> np.ndarray:
        """Optimización para escritura borrosa o desenfocada"""
        # Redimensionar más agresivamente
        height, width = image.shape
        scale = max(3.0, 1200 / max(height, width))
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Deblur con filtro de realce
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(resized, -1, kernel_sharpen)
        
        # Reducir ruido manteniendo bordes
        denoised = cv2.bilateralFilter(sharpened, 15, 80, 80)
        
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Binarización combinada (Otsu + Adaptativa)
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 4
        )
        combined = cv2.bitwise_and(otsu, adaptive)
        
        # Morfología para conectar caracteres fragmentados
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return result
    
    def _optimize_for_small_symbols(self, image: np.ndarray) -> np.ndarray:
        """Optimización para símbolos matemáticos pequeños"""
        # Redimensionar agresivamente para símbolos pequeños
        height, width = image.shape
        scale = max(4.0, 1500 / max(height, width))
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Reducir ruido con filtro no-local means
        denoised = cv2.fastNlMeansDenoising(resized, None, 10, 7, 21)
        
        # Realzar bordes para símbolos finos
        laplacian = cv2.Laplacian(denoised, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        enhanced = cv2.addWeighted(denoised, 0.7, laplacian, 0.3, 0)
        
        # CLAHE con tiles pequeños para detalles locales
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        contrast_enhanced = clahe.apply(enhanced)
        
        # Binarización con umbral más sensible
        _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfología mínima para preservar detalles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return result
    
    def _optimize_for_math_structures(self, image: np.ndarray) -> np.ndarray:
        """Optimización especializada para estructuras matemáticas complejas (fracciones, potencias, etc.)"""
        # Redimensionar para capturar detalles finos
        height, width = image.shape
        scale = max(3.5, 1400 / max(height, width))
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Preservar estructuras verticales (para fracciones)
        # Filtro de mediana para reducir ruido sin afectar líneas
        median_filtered = cv2.medianBlur(resized, 3)
        
        # Realzar líneas horizontales (barras de fracción) y verticales (raíces, sumatorias)
        # Kernel horizontal para detectar barras de fracción
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        horizontal_lines = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel_horizontal)
        
        # Kernel vertical para detectar elementos como √, ∫, ∑
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        vertical_lines = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel_vertical)
        
        # Combinar líneas detectadas con la imagen original
        lines_combined = cv2.add(horizontal_lines, vertical_lines)
        enhanced_structure = cv2.addWeighted(median_filtered, 0.8, lines_combined, 0.2, 0)
        
        # CLAHE adaptativo para mejorar contraste local
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        contrast_enhanced = clahe.apply(enhanced_structure)
        
        # Binarización híbrida para preservar tanto texto fino como estructuras gruesas
        # Otsu para umbral global
        _, otsu_binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptativa para detalles locales
        adaptive_binary = cv2.adaptiveThreshold(
            contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Combinar ambos métodos para obtener lo mejor de ambos
        combined_binary = cv2.bitwise_and(otsu_binary, adaptive_binary)
        
        # Operaciones morfológicas cuidadosas para conectar elementos sin perder estructura
        # Conectar caracteres rotos pero preservar separación entre numerador/denominador
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 1))  # Horizontal delgado
        connected = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel_connect)
        
        # Limpiar ruido muy pequeño
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        cleaned = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_clean)
        
        return cleaned
    
    def _optimize_for_weak_writing(self, image: np.ndarray) -> np.ndarray:
        """Optimización para escritura débil o con poco contraste"""
        # Redimensionar
        height, width = image.shape
        scale = max(2.5, 1000 / max(height, width))
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Mejora agresiva de contraste
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(6, 6))
        enhanced = clahe.apply(resized)
        
        # Gamma correction para realzar trazos débiles
        gamma = 0.6  # Hacer más oscura la imagen
        gamma_corrected = np.array(255 * (enhanced / 255) ** gamma, dtype=np.uint8)
        
        # Filtro de mediana para reducir ruido
        median_filtered = cv2.medianBlur(gamma_corrected, 3)
        
        # Binarización con umbral bajo para capturar trazos débiles
        adaptive = cv2.adaptiveThreshold(
            median_filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 21, 15
        )
        
        # Morfología para fortalecer líneas débiles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        result = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        return result
    
    def _optimize_for_noisy_writing(self, image: np.ndarray) -> np.ndarray:
        """Optimización para escritura con mucho ruido de fondo"""
        # Redimensionar
        height, width = image.shape
        scale = max(2.0, 900 / max(height, width))
        new_size = (int(width * scale), int(height * scale))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # Denoising agresivo con Non-local Means
        denoised = cv2.fastNlMeansDenoising(resized, None, 15, 7, 21)
        
        # Filtro bilateral múltiple para suavizado avanzado
        bilateral1 = cv2.bilateralFilter(denoised, 9, 75, 75)
        bilateral2 = cv2.bilateralFilter(bilateral1, 9, 75, 75)
        
        # CLAHE moderado para no amplificar ruido
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral2)
        
        # Binarización robusta contra ruido
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Operaciones morfológicas para limpiar ruido
        # Opening para eliminar ruido pequeño
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel_clean)
        
        # Closing para conectar líneas rotas
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect)
        
        return result

class EnhancedOCREngine:
    """Motor OCR mejorado con múltiples estrategias de reconocimiento"""
    
    def __init__(self):
        self.reader = None
        self.preprocessor = AdvancedImagePreprocessor()
        self.math_symbol_corrections = {
            # Correcciones comunes de símbolos matemáticos
            'u': '∪', 'U': '∪',  # Unión
            'n': '∩', 'N': '∩',  # Intersección  
            'c': '⊆', 'C': '⊆',  # Subconjunto
            '3': '⊇', 'ᗐ': '⊇',  # Superconjunto
            'e': '∈', 'E': '∈',  # Pertenece
            'o': '∅', 'O': '∅',  # Conjunto vacío
            '|': '∣',  # Tal que
            'x': '×',  # Producto
            'inf': '∞', 'oo': '∞',  # Infinito
            '+-': '±',  # Más-menos
            '<=': '≤', '>=': '≥',  # Desigualdades
            '!=': '≠', '/=': '≠',  # No igual
            '=>': '⇒', '->': '→',  # Implicación
            '<=>': '⇔', '<->': '↔',  # Doble implicación
            
            # Nuevas correcciones para elementos matemáticos avanzados
            
            # Puntos suspensivos y secuencias
            '...': '⋯', '..': '⋯', '. . .': '⋯', '. ..': '⋯', '.. .': '⋯',
            'cdots': '⋯', 'ldots': '…', 'dots': '⋯',
            
            # Sumatorias y productos
            'sum': '∑', 'Sum': '∑', 'SUM': '∑', 'E': '∑',
            'prod': '∏', 'Prod': '∏', 'PROD': '∏', 'II': '∏',
            
            # Raíces
            'sqrt': '√', 'v': '√', 'V': '√', '/': '√',
            'root': '√', 'radical': '√',
            
            # Potencias comunes
            '^2': '²', '^3': '³', '^4': '⁴', '^5': '⁵',
            '2': '²', '3': '³', # Solo cuando están en posición de exponente
            
            # Fracciones
            '/': '/', '÷': '/', 'over': '/',
            
            # Integrales
            'int': '∫', 'integral': '∫', 'f': '∫', 'J': '∫',
            
            # Límites
            'lim': 'lim', 'limit': 'lim',
            
            # Derivadas
            'd/dx': '∂', 'partial': '∂', 'del': '∂',
            
            # Otros símbolos matemáticos
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ',
            'epsilon': 'ε', 'theta': 'θ', 'lambda': 'λ', 'mu': 'μ',
            'pi': 'π', 'sigma': 'σ', 'tau': 'τ', 'phi': 'φ',
            'omega': 'ω',
        }
        
    def initialize_reader(self):
        """Inicializar EasyOCR de forma segura"""
        if self.reader is None:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False, download_enabled=False)
                print("✅ EasyOCR inicializado correctamente")
                return True
            except Exception as e:
                print(f"❌ Error inicializando EasyOCR: {e}")
                return False
        return True
    
    def recognize_with_multiple_strategies(self, image: np.ndarray) -> Dict[str, any]:
        """Reconoce texto usando múltiples estrategias y selecciona el mejor resultado"""
        
        if not self.initialize_reader():
            return {"text": "", "confidence": 0.0, "method": "error"}
        
        # NUEVA ESTRATEGIA: Análisis espacial directo primero
        if SPATIAL_ANALYZER_AVAILABLE:
            print("🎯 Intentando análisis espacial directo...")
            try:
                spatial_result = self._try_spatial_analysis_direct(image)
                if spatial_result and spatial_result["text"].strip():
                    print(f"✅ Análisis espacial exitoso: '{spatial_result['text']}'")
                    return spatial_result
            except Exception as e:
                print(f"⚠️ Error en análisis espacial directo: {e}")
        
        # Crear múltiples variantes de la imagen
        variants = self.preprocessor.create_multiple_variants(image)
        
        # Configuraciones de EasyOCR para diferentes casos
        ocr_configs = [
            {"width_ths": 0.1, "height_ths": 0.1, "detail": 1, "paragraph": False},  # Muy sensible para símbolos diminutos
            {"width_ths": 0.2, "height_ths": 0.2, "detail": 1, "paragraph": False},  # Símbolos pequeños
            {"width_ths": 0.5, "height_ths": 0.5, "detail": 1, "paragraph": False},  # Texto normal
            {"width_ths": 0.7, "height_ths": 0.7, "detail": 1, "paragraph": False},  # Texto grande
            {"detail": 0, "paragraph": False},  # Modo simple sin detalles
            # Configuraciones especiales para matemáticas complejas
            {"width_ths": 0.3, "height_ths": 0.1, "detail": 1, "paragraph": False},  # Fracciones (ancho normal, alto pequeño)
            {"width_ths": 0.1, "height_ths": 0.5, "detail": 1, "paragraph": False},  # Elementos verticales (∑, ∫, √)
        ]
        
        all_results = []
        
        print(f"🔍 Probando {len(variants)} variantes de imagen con {len(ocr_configs)} configuraciones OCR...")
        
        for variant_name, variant_image in variants:
            print(f"📝 Procesando variante: {variant_name}")
            
            for config_idx, config in enumerate(ocr_configs):
                try:
                    start_time = time.time()
                    
                    # Aplicar timeout de 20 segundos por intento
                    results = self.reader.readtext(variant_image, **config)
                    
                    elapsed = time.time() - start_time
                    if elapsed > 20:
                        print(f"⚠️ Timeout en {variant_name} config {config_idx}")
                        continue
                    
                    # Procesar resultados con análisis espacial si está disponible
                    if results and SPATIAL_ANALYZER_AVAILABLE:
                        try:
                            # Usar análisis espacial para interpretar mejor los resultados
                            spatial_analyzer = SpatialMathAnalyzer()
                            spatial_text = spatial_analyzer.analyze_spatial_math(results)
                            
                            if spatial_text.strip():
                                # Calcular métricas para análisis espacial
                                confidences = [conf for _, _, conf in results if conf > 0.05]
                                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                                math_score = self._calculate_math_score(spatial_text)
                                
                                result = {
                                    "text": spatial_text,
                                    "confidence": avg_confidence,
                                    "math_score": math_score,
                                    "combined_score": (avg_confidence * 0.4) + (math_score * 0.6),  # Priorizar math_score
                                    "method": f"{variant_name}_config{config_idx}_spatial",
                                    "processing_time": elapsed,
                                    "spatial_analysis": True
                                }
                                
                                all_results.append(result)
                                print(f"   🎯 Espacial: '{spatial_text}' (conf: {avg_confidence:.2f}, math: {math_score:.2f})")
                        except Exception as spatial_error:
                            print(f"⚠️ Error en análisis espacial para {variant_name}: {spatial_error}")
                    
                    # Procesamiento estándar como fallback
                    if results:
                        texts = []
                        confidences = []
                        
                        for (bbox, text, confidence) in results:
                            if confidence > 0.05:  # Umbral muy bajo para matemáticas
                                texts.append(text)
                                confidences.append(confidence)
                        
                        if texts:
                            combined_text = ' '.join(texts)
                            avg_confidence = sum(confidences) / len(confidences)
                            math_score = self._calculate_math_score(combined_text)
                            
                            result = {
                                "text": combined_text,
                                "confidence": avg_confidence,
                                "math_score": math_score,
                                "combined_score": (avg_confidence * 0.6) + (math_score * 0.4),
                                "method": f"{variant_name}_config{config_idx}_standard",
                                "processing_time": elapsed,
                                "spatial_analysis": False
                            }
                            
                            all_results.append(result)
                            print(f"   📋 Estándar: '{combined_text}' (conf: {avg_confidence:.2f}, math: {math_score:.2f})")
                
                except Exception as e:
                    print(f"⚠️ Error en {variant_name} config {config_idx}: {e}")
                    continue
        
        # Seleccionar el mejor resultado
        if all_results:
            # Priorizar resultados con análisis espacial
            spatial_results = [r for r in all_results if r.get("spatial_analysis", False)]
            standard_results = [r for r in all_results if not r.get("spatial_analysis", False)]
            
            if spatial_results:
                best_result = max(spatial_results, key=lambda x: x["combined_score"])
                print(f"🎯 Mejor resultado (espacial): '{best_result['text']}' (score: {best_result['combined_score']:.2f})")
            else:
                best_result = max(standard_results, key=lambda x: x["combined_score"])
                print(f"📋 Mejor resultado (estándar): '{best_result['text']}' (score: {best_result['combined_score']:.2f})")
            
            # Aplicar correcciones finales solo si no usó análisis espacial
            if not best_result.get("spatial_analysis", False):
                corrected_text = self._apply_math_corrections(best_result["text"])
                best_result["text"] = corrected_text
                best_result["corrected"] = True
            
            return best_result
        else:
            print("❌ No se pudo reconocer ningún texto")
            return {"text": "", "confidence": 0.0, "method": "no_results"}
    
    def _try_spatial_analysis_direct(self, image: np.ndarray) -> Optional[Dict]:
        """Intenta análisis espacial directo en la imagen original"""
        try:
            # Usar configuración óptima para análisis espacial
            config = {"width_ths": 0.2, "height_ths": 0.2, "detail": 1, "paragraph": False}
            
            # OCR directo con detalles de posición
            results = self.reader.readtext(image, **config)
            
            if not results:
                return None
            
            # Aplicar análisis espacial
            spatial_analyzer = SpatialMathAnalyzer()
            spatial_text = spatial_analyzer.analyze_spatial_math(results)
            
            if not spatial_text.strip():
                return None
            
            # Calcular métricas
            confidences = [conf for _, _, conf in results if conf > 0.05]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            math_score = self._calculate_math_score(spatial_text)
            
            return {
                "text": spatial_text,
                "confidence": avg_confidence,
                "math_score": math_score,
                "combined_score": (avg_confidence * 0.3) + (math_score * 0.7),  # Priorizar fuertemente math_score
                "method": "direct_spatial_analysis",
                "processing_time": 0.0,
                "spatial_analysis": True
            }
            
        except Exception as e:
            print(f"Error en análisis espacial directo: {e}")
            return None
    
    def _calculate_math_score(self, text: str) -> float:
        """Calcula un puntaje de calidad matemática del texto"""
        if not text or not text.strip():
            return 0.0
        
        score = 0.0
        text_clean = text.strip()
        
        # Símbolos matemáticos comunes
        math_symbols = ['∪', '∩', '⊆', '⊇', '⊃', '⊂', '∈', '∉', '∅', '=', '+', '-', '*', '/', '^', '_']
        for symbol in math_symbols:
            if symbol in text_clean:
                score += 1.0
        
        # Elementos matemáticos avanzados (puntuación mayor)
        advanced_elements = [
            '∑', '∏', '∫', '√', '⋯', '…',  # Sumatorias, productos, integrales, raíces, puntos
            '²', '³', '⁴', '⁵',  # Potencias
            'lim', 'sin', 'cos', 'tan', 'log', 'ln',  # Funciones
            '∂', '∆', '∇',  # Derivadas y operadores
        ]
        for element in advanced_elements:
            if element in text_clean:
                score += 2.0  # Puntuación doble para elementos avanzados
        
        # Patrones de secuencias matemáticas
        import re
        
        # Detectar secuencias como "1+2+3+...+n" o "1²+2²+3²+...+n²"
        sequence_patterns = [
            r'\d+\+\d+\+\d+\+[⋯…\.]+\+\w+',  # 1+2+3+...+n
            r'\d+[²³⁴⁵]\+\d+[²³⁴⁵]\+\d+[²³⁴⁵]\+[⋯…\.]+\+\w+[²³⁴⁵]',  # 1²+2²+3²+...+n²
            r'\d+\*\d+\*\d+\*[⋯…\.]+\*\w+',  # 1*2*3*...*n
            r'[a-zA-Z]\d+\+[a-zA-Z]\d+\+[a-zA-Z]\d+\+[⋯…\.]+',  # a₁+a₂+a₃+...
        ]
        
        for pattern in sequence_patterns:
            if re.search(pattern, text_clean):
                score += 3.0  # Puntuación triple para secuencias
        
        # Detectar fracciones
        fraction_patterns = [
            r'\d+/\d+',  # Fracciones simples como 1/2
            r'[a-zA-Z]+/[a-zA-Z]+',  # Fracciones algebraicas como a/b
            r'\([^)]+\)/\([^)]+\)',  # Fracciones complejas como (x+1)/(x-1)
        ]
        
        for pattern in fraction_patterns:
            if re.search(pattern, text_clean):
                score += 2.0
        
        # Detectar sumatorias y productos con límites
        summation_patterns = [
            r'∑.*=.*',  # ∑ con límites
            r'∏.*=.*',  # ∏ con límites
            r'∫.*d[a-zA-Z]',  # Integrales con diferencial
        ]
        
        for pattern in summation_patterns:
            if re.search(pattern, text_clean):
                score += 3.0
        
        # Variables matemáticas (letras aisladas)
        if re.search(r'\b[A-Za-z]\b', text_clean):
            score += 0.5
        
        # Números
        if re.search(r'\d', text_clean):
            score += 0.3
        
        # Penalización por caracteres raros
        weird_chars = ['#', '$', '%', '&', '@']
        for char in weird_chars:
            if char in text_clean:
                score -= 0.5
        
        # Bonus por longitud apropiada para matemáticas
        if 2 <= len(text_clean) <= 50:  # Aumentado para fórmulas más complejas
            score += 0.5
        
        # Normalizar score (aumentado el divisor para elementos avanzados)
        return max(0.0, min(1.0, score / 10.0))
    
    def _apply_math_corrections(self, text: str) -> str:
        """Aplica correcciones específicas para símbolos matemáticos"""
        corrected = text
        
        # Aplicar diccionario de correcciones básicas
        for wrong, right in self.math_symbol_corrections.items():
            corrected = corrected.replace(wrong, right)
        
        # Correcciones avanzadas con patrones regex
        import re
        
        # 1. Corregir puntos suspensivos en secuencias
        # Patrones como ". . .", "...", "..", "• • •" → "⋯"
        corrected = re.sub(r'\.{2,}', '⋯', corrected)
        corrected = re.sub(r'\.\s+\.\s+\.', '⋯', corrected)
        corrected = re.sub(r'•\s*•\s*•', '⋯', corrected)
        corrected = re.sub(r'°\s*°\s*°', '⋯', corrected)
        
        # 2. Detectar y corregir potencias
        # Números pequeños después de letras/números → exponentes
        corrected = re.sub(r'([a-zA-Z0-9])\s*([²³⁴⁵])', r'\1\2', corrected)
        corrected = re.sub(r'([a-zA-Z0-9])\^([0-9])', r'\1^{\2}', corrected)
        
        # Casos especiales de potencias mal reconocidas
        corrected = re.sub(r'([a-zA-Z0-9])\s*2(?=\s|$|\+|\-|\*|\/)', r'\1²', corrected)
        corrected = re.sub(r'([a-zA-Z0-9])\s*3(?=\s|$|\+|\-|\*|\/)', r'\1³', corrected)
        
        # 3. Corregir sumatorias y productos
        # Patrones comunes mal reconocidos
        corrected = re.sub(r'\bE\b(?=.*=)', '∑', corrected)  # E → ∑ si hay límites
        corrected = re.sub(r'\bΣ\b', '∑', corrected)  # Σ mayúscula → ∑
        corrected = re.sub(r'\bII\b', '∏', corrected)  # II → ∏
        corrected = re.sub(r'\bΠ\b', '∏', corrected)  # Π mayúscula → ∏
        
        # 4. Corregir integrales
        corrected = re.sub(r'\bf\b(?=.*d[a-zA-Z])', '∫', corrected)  # f → ∫ si hay diferencial
        corrected = re.sub(r'\bJ\b(?=.*d[a-zA-Z])', '∫', corrected)  # J → ∫ si hay diferencial
        corrected = re.sub(r'\b∫\b', '∫', corrected)  # Normalizar integrales
        
        # 5. Corregir raíces
        corrected = re.sub(r'\bv\s*([0-9a-zA-Z])', r'√\1', corrected)  # v → √
        corrected = re.sub(r'\bV\s*([0-9a-zA-Z])', r'√\1', corrected)  # V → √
        corrected = re.sub(r'√\s+', '√', corrected)  # Eliminar espacios después de √
        
        # 6. Mejorar detección de fracciones
        # Espacios alrededor de barras de fracción
        corrected = re.sub(r'([0-9a-zA-Z\)])\s*/\s*([0-9a-zA-Z\(])', r'\1/\2', corrected)
        
        # 7. Corregir secuencias numéricas comunes
        # Patrones como "1 + 2 + 3 + ... + n"
        corrected = re.sub(r'(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*\+\s*[⋯…\.]+\s*\+\s*([a-zA-Z])', 
                          r'\1+\2+\3+⋯+\4', corrected)
        
        # Secuencias con potencias: "1² + 2² + 3² + ... + n²"
        corrected = re.sub(r'(\d+)[²³⁴⁵]\s*\+\s*(\d+)[²³⁴⁵]\s*\+\s*(\d+)[²³⁴⁵]\s*\+\s*[⋯…\.]+\s*\+\s*([a-zA-Z])[²³⁴⁵]', 
                          r'\1²+\2²+\3²+⋯+\4²', corrected)
        
        # 8. Corregir límites de sumatorias/integrales
        # Patrones como "∑ i=1" → "∑_{i=1}"
        corrected = re.sub(r'∑\s*([a-zA-Z])\s*=\s*([0-9]+)', r'∑_{\1=\2}', corrected)
        corrected = re.sub(r'∏\s*([a-zA-Z])\s*=\s*([0-9]+)', r'∏_{\1=\2}', corrected)
        
        # 9. Corregir funciones trigonométricas y logarítmicas
        trig_functions = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan']
        log_functions = ['log', 'ln', 'lg']
        
        for func in trig_functions + log_functions:
            # Asegurar espaciado correcto
            corrected = re.sub(f'\\b{func}\\s+', f'{func} ', corrected)
        
        # 10. Corregir letras griegas comunes mal reconocidas
        greek_corrections = {
            'a': 'α', 'B': 'β', 'y': 'γ', 'δ': 'δ', 'ε': 'ε',
            '0': 'θ', 'λ': 'λ', 'μ': 'μ', 'π': 'π', 'σ': 'σ',
            'τ': 'τ', 'φ': 'φ', 'ω': 'ω'
        }
        
        # Solo aplicar si están en contexto matemático (rodeadas de símbolos/números)
        for latin, greek in greek_corrections.items():
            pattern = f'(?<=[0-9+\\-*/=])\\s*{re.escape(latin)}\\s*(?=[0-9+\\-*/=])'
            corrected = re.sub(pattern, f' {greek} ', corrected)
        
        # 11. Limpiar espaciado
        # Espacios alrededor de operadores principales
        corrected = re.sub(r'\s*([+\-*/=<>≤≥≠])\s*', r' \1 ', corrected)
        
        # Espacios alrededor de símbolos de conjuntos
        corrected = re.sub(r'\s*([∪∩⊆⊇⊃⊂∈∉])\s*', r' \1 ', corrected)
        
        # Sin espacios en elementos unitarios
        corrected = re.sub(r'([a-zA-Z0-9])\s+([²³⁴⁵])', r'\1\2', corrected)  # x ² → x²
        corrected = re.sub(r'√\s+([a-zA-Z0-9])', r'√\1', corrected)  # √ x → √x
        
        # 12. Limpiar espacios múltiples
        corrected = re.sub(r'\s+', ' ', corrected).strip()
        
        # 13. Correcciones finales específicas
        # Casos especiales que aparecen frecuentemente
        final_corrections = {
            '∑ ∞': '∑_{i=1}^∞',
            '∏ ∞': '∏_{i=1}^∞',
            '∫ ∞': '∫_0^∞',
            'lim ∞': 'lim_{n→∞}',
            '→ ∞': '→ ∞',
            '∞ →': '→ ∞',
        }
        
        for wrong, right in final_corrections.items():
            corrected = corrected.replace(wrong, right)
        
        return corrected

# Función principal de interfaz
def enhanced_ocr_recognition(image_path: str) -> Dict[str, any]:
    """
    Función principal para reconocimiento OCR mejorado
    """
    try:
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"No se pudo cargar la imagen: {image_path}"}
        
        # Inicializar motor OCR
        ocr_engine = EnhancedOCREngine()
        
        # Realizar reconocimiento
        result = ocr_engine.recognize_with_multiple_strategies(image)
        
        return result
        
    except Exception as e:
        return {"error": f"Error en reconocimiento OCR: {str(e)}"}

if __name__ == "__main__":
    # Función de prueba
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = enhanced_ocr_recognition(image_path)
        
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"✅ Texto reconocido: '{result['text']}'")
            print(f"📊 Confianza: {result['confidence']:.2f}")
            print(f"🧮 Puntaje matemático: {result['math_score']:.2f}")
            print(f"🎯 Método usado: {result['method']}")
    else:
        print("Uso: python enhanced_ocr_system.py <ruta_imagen>")
