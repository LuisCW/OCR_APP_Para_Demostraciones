"""
Configuración mejorada específica para símbolos matemáticos manuscritos
Optimizado para casos como "B ⊇ A ∪ B" y otros símbolos de conjuntos
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class MathSymbolPreprocessor:
    """Preprocesamiento especializado para símbolos matemáticos manuscritos"""
    
    def __init__(self):
        self.debug_mode = True
        
    def enhance_math_symbols(self, image):
        """Mejora específica para símbolos matemáticos manuscritos"""
        
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Redimensionar significativamente para mejor reconocimiento
        height, width = gray.shape
        scale_factor = max(3.0, 1000 / max(height, width))  # Mínimo 3x, máximo para llegar a 1000px
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 2. Reducir ruido con filtro bilateral preservando bordes
        denoised = cv2.bilateralFilter(resized, 15, 80, 80)
        
        # 3. Mejorar contraste con CLAHE adaptativo
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 4. Filtro de realce de bordes suave
        kernel_sharpen = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen * 0.3)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # 5. Binarización adaptativa mejorada
        # Primero Otsu para encontrar umbral óptimo
        _, otsu_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Luego binarización adaptativa para detalles finos
        adaptive_thresh = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )
        
        # Combinar ambos métodos (tomar el mejor de cada píxel)
        combined = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
        
        # 6. Operaciones morfológicas para conectar trazos de símbolos
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_connect)
        
        # 7. Limpiar ruido pequeño
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_clean)
        
        if self.debug_mode:
            self.save_debug_steps(gray, resized, enhanced, sharpened, combined, cleaned)
        
        return cleaned
    
    def save_debug_steps(self, original, resized, enhanced, sharpened, binary, final):
        """Guardar pasos del preprocesamiento para depuración"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title('1. Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(resized, cmap='gray')
            axes[0, 1].set_title('2. Redimensionado')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(enhanced, cmap='gray')
            axes[0, 2].set_title('3. CLAHE')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(sharpened, cmap='gray')
            axes[1, 0].set_title('4. Realzado')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(binary, cmap='gray')
            axes[1, 1].set_title('5. Binarizado')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(final, cmap='gray')
            axes[1, 2].set_title('6. Final')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig('math_preprocessing_steps.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # También guardar la imagen final para uso directo
            cv2.imwrite('math_preprocessed_final.png', final)
            
        except Exception as e:
            print(f"Error guardando debug: {e}")

class MathOCROptimizer:
    """Optimizador específico para OCR de símbolos matemáticos"""
    
    def __init__(self):
        self.math_symbols_whitelist = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789"
            "+-=()[]{}.,;:!?"
            "∪∩⊇⊆⊃⊂∈∉⋃⋂∅∞∂∫∮∑∏√∝∀∃∧∨¬→↔⇒⇔≡≠≤≥≪≫"
            "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
            " "  # Espacios importantes
        )
        
        self.tesseract_configs = [
            # Configuraciones específicas para símbolos matemáticos
            '--psm 8 --oem 3',  # Una sola palabra
            '--psm 7 --oem 3',  # Una línea de texto
            '--psm 13 --oem 3',  # Raw line, tratamiento mínimo
            f'--psm 8 --oem 3 -c tessedit_char_whitelist={self.math_symbols_whitelist}',
            f'--psm 7 --oem 3 -c tessedit_char_whitelist={self.math_symbols_whitelist}',
            '--psm 6 --oem 3 -c tessedit_char_blacklist=|',  # Excluir caracteres problemáticos
            '--psm 8 --oem 1',  # LSTM OCR Engine
            '--psm 8 --oem 2',  # Legacy + LSTM
        ]
    
    def optimize_for_math_symbols(self, image, ocr_function):
        """
        Probar múltiples configuraciones y seleccionar el mejor resultado
        para símbolos matemáticos
        """
        results = []
        
        for config in self.tesseract_configs:
            try:
                result = ocr_function(image, config)
                confidence = self.calculate_math_confidence(result)
                results.append((result, confidence, config))
            except Exception as e:
                print(f"Error con config '{config}': {e}")
                continue
        
        if not results:
            return ""
        
        # Ordenar por confianza y devolver el mejor
        results.sort(key=lambda x: x[1], reverse=True)
        best_result, best_confidence, best_config = results[0]
        
        print(f"Mejor resultado: '{best_result}' (confianza: {best_confidence:.2f}, config: '{best_config}')")
        
        return best_result
    
    def calculate_math_confidence(self, text):
        """
        Calcular un puntaje de confianza para texto matemático
        """
        if not text or not text.strip():
            return 0.0
        
        confidence = 0.0
        text = text.strip()
        
        # Puntos por contener símbolos matemáticos comunes
        math_symbols = ['∪', '∩', '⊇', '⊆', '⊃', '⊂', '∈', '∉', '=', '+', '-']
        for symbol in math_symbols:
            if symbol in text:
                confidence += 1.0
        
        # Puntos por estructura típica de expresiones matemáticas
        if any(char in text for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            confidence += 0.5  # Variables mayúsculas
        
        if any(char in text for char in 'abcdefghijklmnopqrstuvwxyz'):
            confidence += 0.3  # Variables minúsculas
        
        # Penalización por caracteres extraños
        strange_chars = ['|', '_', '^', '~', '`']
        for char in strange_chars:
            if char in text:
                confidence -= 0.5
        
        # Bonus por longitud razonable
        if 3 <= len(text) <= 20:
            confidence += 0.5
        
        # Penalización por texto muy corto o muy largo
        if len(text) < 2:
            confidence -= 1.0
        if len(text) > 30:
            confidence -= 1.0
        
        return max(0.0, confidence)

def test_math_preprocessing():
    """Función de prueba para el preprocesamiento matemático"""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    
    print("Creando imagen de prueba con símbolos matemáticos...")
    
    # Crear imagen de prueba
    img = Image.new('RGB', (600, 200), 'white')
    draw = ImageDraw.Draw(img)
    
    # Intentar usar una fuente del sistema
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    # Escribir expresión matemática
    text = "B ⊇ A ∪ B"
    draw.text((50, 75), text, font=font, fill='black')
    
    # Convertir a numpy array para procesamiento
    img_array = np.array(img)
    
    # Añadir algo de ruido para simular escritura manuscrita
    noise = np.random.normal(0, 15, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Aplicar preprocesamiento
    preprocessor = MathSymbolPreprocessor()
    processed = preprocessor.enhance_math_symbols(img_array)
    
    print("✓ Preprocesamiento completado")
    print("✓ Archivos de debug guardados:")
    print("  - math_preprocessing_steps.png")
    print("  - math_preprocessed_final.png")
    
    return processed

if __name__ == "__main__":
    print("Probando preprocesamiento específico para símbolos matemáticos...")
    test_math_preprocessing()
    print("Prueba completada. Revise los archivos generados.")
