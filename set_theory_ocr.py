"""
Preprocesador especializado para manuscritos de teoría de conjuntos
Optimizado para EasyOCR y símbolos matemáticos manuscritos
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class SetTheoryPreprocessor:
    """Preprocesador especializado para teoría de conjuntos manuscrita"""
    
    def __init__(self):
        self.debug_mode = True
        
        # Símbolos comunes en teoría de conjuntos
        self.set_symbols = [
            '∪', '∩', '⊆', '⊇', '⊂', '⊃', '∈', '∉', '∅', '∖', 
            '⊕', '⊗', '∁', '℘', '∆', '⊥', '⊤'
        ]
        
        # Caracteres que suelen confundirse
        self.confusion_map = {
            'U': '∪',  # U mayúscula → unión
            'n': '∩',  # n minúscula → intersección
            'c': '⊂',  # c minúscula → subconjunto propio
            'C': '⊃',  # C mayúscula → superconjunto
            'e': '∈',  # e minúscula → pertenece
            'O': '∅',  # O mayúscula → conjunto vacío
            '0': '∅',  # cero → conjunto vacío
        }
    
    def enhance_handwritten_math(self, image):
        """Mejora específica para matemáticas manuscritas"""
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. SUPER-RESOLUCIÓN: Aumentar tamaño 4x para mejor detalle
        height, width = gray.shape
        target_height = max(800, height * 4)  # Mínimo 800px de alto
        scale = target_height / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Usar interpolación Lanczos para mejor calidad en texto
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. REDUCCIÓN DE RUIDO preservando bordes
        denoised = cv2.bilateralFilter(resized, 15, 100, 100)
        
        # 3. MEJORA DE CONTRASTE adaptativa
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 4. REALCE DE BORDES suave para símbolos
        gaussian_blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        unsharp_mask = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
        unsharp_mask = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
        
        # 5. BINARIZACIÓN híbrida optimizada
        # Método 1: Otsu global
        _, otsu_binary = cv2.threshold(unsharp_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Método 2: Adaptativa local para detalles finos
        adaptive_binary = cv2.adaptiveThreshold(
            unsharp_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 8
        )
        
        # Combinar ambos métodos: usar el que da mejor contraste por región
        combined = cv2.bitwise_and(otsu_binary, adaptive_binary)
        
        # 6. REFINAMIENTO morfológico específico para texto manuscrito
        # Conectar trazos de símbolos que pueden estar separados
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_connect)
        
        # Limpiar ruido muy pequeño sin afectar puntos y acentos
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        cleaned = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_clean)
        
        # 7. POST-PROCESAMIENTO para símbolos específicos
        final = self.enhance_specific_symbols(cleaned)
        
        if self.debug_mode:
            self.save_preprocessing_steps(gray, resized, enhanced, combined, final)
        
        return final
    
    def enhance_specific_symbols(self, binary_image):
        """Post-procesamiento específico para símbolos de conjuntos"""
        
        # Encontrar contornos para analizar formas
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen resultado
        result = binary_image.copy()
        
        for contour in contours:
            # Calcular propiedades del contorno
            area = cv2.contourArea(contour)
            
            # Filtrar contornos muy pequeños (ruido) o muy grandes (fondo)
            if area < 50 or area > binary_image.shape[0] * binary_image.shape[1] * 0.3:
                continue
            
            # Obtener bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Mejorar símbolos específicos basándose en forma
            aspect_ratio = w / h if h > 0 else 0
            
            # Círculos o símbolos redondos (∅, ∘, etc.)
            if 0.7 <= aspect_ratio <= 1.3 and area > 100:
                # Suavizar contornos circulares
                epsilon = 0.02 * cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(contour, epsilon, True)
                cv2.fillPoly(result, [smoothed], 255)
            
            # Símbolos alargados horizontalmente (⊆, ⊇, etc.)
            elif aspect_ratio > 1.2:
                # Reforzar símbolos de contención
                cv2.rectangle(result, (x-1, y-1), (x+w+1, y+h+1), 255, 1)
        
        return result
    
    def save_preprocessing_steps(self, original, resized, enhanced, binary, final):
        """Guardar pasos del preprocesamiento para debug"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Preprocesamiento para Teoría de Conjuntos Manuscrita', fontsize=16)
            
            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title('1. Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(resized, cmap='gray')
            axes[0, 1].set_title('2. Super-resolución 4x')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(enhanced, cmap='gray')
            axes[0, 2].set_title('3. Contraste + Realce')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(binary, cmap='gray')
            axes[1, 0].set_title('4. Binarización Híbrida')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(final, cmap='gray')
            axes[1, 1].set_title('5. Final Optimizado')
            axes[1, 1].axis('off')
            
            # Histograma de la imagen final
            axes[1, 2].hist(final.ravel(), bins=50, alpha=0.7)
            axes[1, 2].set_title('Histograma Final')
            axes[1, 2].set_xlabel('Intensidad')
            axes[1, 2].set_ylabel('Frecuencia')
            
            plt.tight_layout()
            plt.savefig('debug_set_theory_preprocessing.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Guardar imagen final procesada
            cv2.imwrite('debug_set_theory_final.png', final)
            
            print("🖼️  Debug guardado: debug_set_theory_preprocessing.png")
            print("🖼️  Imagen final: debug_set_theory_final.png")
            
        except Exception as e:
            print(f"⚠️  Error guardando debug: {e}")


class EasyOCROptimizer:
    """Optimizador específico para EasyOCR con manuscritos matemáticos"""
    
    def __init__(self):
        self.preprocessor = SetTheoryPreprocessor()
        
        # Configuraciones optimizadas para diferentes tipos de manuscritos
        self.ocr_configs = {
            'set_theory': {
                'width_ths': 0.3,      # Detectar símbolos estrechos
                'height_ths': 0.3,     # Detectar símbolos pequeños
                'detail': 1,           # Incluir coordenadas y confianza
                'paragraph': False,    # No agrupar en párrafos
                'min_size': 10,        # Tamaño mínimo de detección
                'text_threshold': 0.1, # Umbral bajo para símbolos débiles
                'link_threshold': 0.1, # Umbral bajo para conectar caracteres
                'low_text': 0.1,       # Detectar texto de bajo contraste
                'slope_ths': 0.3,      # Tolerancia a inclinación
                'ycenter_ths': 0.7,    # Tolerancia vertical
                'height_ths': 0.7,     # Tolerancia de altura
                'width_ths': 0.9,      # Tolerancia de ancho
                'add_margin': 0.2      # Margen adicional
            },
            'conservative': {
                'width_ths': 0.8,
                'height_ths': 0.8,
                'detail': 1,
                'paragraph': False,
                'min_size': 20
            },
            'aggressive': {
                'width_ths': 0.1,
                'height_ths': 0.1,
                'detail': 1,
                'paragraph': False,
                'min_size': 5
            }
        }
    
    def recognize_set_theory_manuscript(self, image):
        """Reconocimiento optimizado para manuscritos de teoría de conjuntos"""
        
        try:
            import easyocr
            
            # Crear reader optimizado
            reader = easyocr.Reader(['en', 'es'], gpu=False, verbose=False)
            
            # Preprocesar imagen
            processed_image = self.preprocessor.enhance_handwritten_math(image)
            
            print("🔍 Analizando manuscrito de teoría de conjuntos...")
            
            results = []
            confidences = []
            
            # Probar diferentes configuraciones
            for config_name, config_params in self.ocr_configs.items():
                try:
                    print(f"   🧪 Probando configuración: {config_name}")
                    
                    # Ejecutar OCR con configuración específica
                    ocr_results = reader.readtext(processed_image, **config_params)
                    
                    if ocr_results:
                        for bbox, text, confidence in ocr_results:
                            if confidence > 0.05 and text.strip():  # Umbral muy bajo
                                results.append((text.strip(), confidence, config_name))
                                confidences.append(confidence)
                                print(f"      📝 '{text}' (conf: {confidence:.3f})")
                    
                except Exception as e:
                    print(f"      ❌ Error en configuración {config_name}: {e}")
            
            if not results:
                print("   ⚠️  No se detectó texto")
                return ""
            
            # Seleccionar mejor resultado
            best_result = self.select_best_result(results)
            
            # Post-procesar para corregir símbolos comunes
            corrected_result = self.correct_common_mistakes(best_result)
            
            print(f"   🏆 Mejor resultado: '{corrected_result}'")
            
            return corrected_result
            
        except ImportError:
            print("❌ EasyOCR no está instalado")
            return ""
        except Exception as e:
            print(f"❌ Error en reconocimiento: {e}")
            return ""
    
    def select_best_result(self, results):
        """Seleccionar el mejor resultado basado en contenido matemático"""
        
        if not results:
            return ""
        
        # Puntuar cada resultado
        scored_results = []
        
        for text, confidence, config in results:
            score = confidence * 100  # Base: confianza del OCR
            
            # Bonus por símbolos matemáticos reconocidos
            math_symbols = ['∪', '∩', '⊆', '⊇', '⊂', '⊃', '∈', '∉', '∅']
            for symbol in math_symbols:
                if symbol in text:
                    score += 50
            
            # Bonus por letras matemáticas comunes
            if any(char in text.upper() for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                score += 20
            
            # Bonus por estructura típica de teoría de conjuntos
            if any(pattern in text.upper() for pattern in ['AU', 'BU', 'AB', 'BA']):
                score += 30
            
            # Penalización por caracteres extraños
            strange_chars = ['|', '_', '^', '~', '#', '$', '%', '@']
            for char in strange_chars:
                if char in text:
                    score -= 10
            
            scored_results.append((text, score, confidence, config))
        
        # Ordenar por puntuación
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver el texto del mejor resultado
        return scored_results[0][0] if scored_results else ""
    
    def correct_common_mistakes(self, text):
        """Corregir errores comunes en reconocimiento de símbolos"""
        
        if not text:
            return text
        
        corrected = text
        
        # Correcciones específicas para teoría de conjuntos
        corrections = {
            # Símbolos de unión
            ' U ': ' ∪ ',
            'U': '∪',
            ' u ': ' ∪ ',
            
            # Símbolos de intersección
            ' n ': ' ∩ ',
            ' ^ ': ' ∩ ',
            
            # Símbolos de subconjunto/superconjunto
            ' c ': ' ⊂ ',
            ' C ': ' ⊃ ',
            '<=': '⊆',
            '>=': '⊇',
            
            # Pertenencia
            ' e ': ' ∈ ',
            ' E ': ' ∈ ',
            
            # Conjunto vacío
            ' O ': ' ∅ ',
            ' 0 ': ' ∅ ',
            'empty': '∅',
            
            # Espaciado común
            'AU': 'A ∪ ',
            'BU': 'B ∪ ',
            'AB': 'A ∩ B',
            'BA': 'B ∩ A',
        }
        
        for mistake, correction in corrections.items():
            corrected = corrected.replace(mistake, correction)
        
        # Limpiar espacios extra
        corrected = ' '.join(corrected.split())
        
        return corrected


def test_set_theory_recognition():
    """Función de prueba para reconocimiento de teoría de conjuntos"""
    
    print("🧪 PROBANDO RECONOCIMIENTO DE TEORÍA DE CONJUNTOS")
    print("=" * 60)
    
    optimizer = EasyOCROptimizer()
    
    # Crear imagen de prueba
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new('RGB', (800, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    # Texto de prueba con símbolos de teoría de conjuntos
    test_text = "B ⊇ A ∪ B"
    draw.text((100, 100), test_text, font=font, fill='black')
    
    # Guardar imagen de prueba
    img.save("test_set_theory.png")
    print("📸 Imagen de prueba creada: test_set_theory.png")
    
    # Convertir a formato OpenCV
    import cv2
    import numpy as np
    
    img_array = np.array(img)
    cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Probar reconocimiento
    result = optimizer.recognize_set_theory_manuscript(cv_image)
    
    print(f"\n📊 RESULTADO:")
    print(f"   Original: {test_text}")
    print(f"   Reconocido: {result}")
    
    # Limpiar archivo de prueba
    import os
    try:
        os.remove("test_set_theory.png")
    except:
        pass
    
    return result


if __name__ == "__main__":
    test_set_theory_recognition()
