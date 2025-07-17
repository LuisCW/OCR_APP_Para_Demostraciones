"""
Analizador espacial para elementos matemáticos manuscritos
Detecta fracciones, potencias y otros elementos basándose en su posición espacial
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import re

class SpatialMathElement:
    """Representa un elemento matemático con información espacial"""
    def __init__(self, text: str, bbox: Tuple[int, int, int, int], confidence: float):
        self.text = text
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.center_x = (self.x1 + self.x2) // 2
        self.center_y = (self.y1 + self.y2) // 2
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.area = self.width * self.height

class SpatialMathAnalyzer:
    """Analizador espacial para elementos matemáticos manuscritos"""
    
    def __init__(self):
        self.elements = []
        self.debug_mode = True
        
    def analyze_spatial_math(self, ocr_results: List[Tuple]) -> str:
        """
        Analiza resultados OCR considerando posiciones espaciales
        ocr_results: Lista de tuplas (bbox, text, confidence)
        """
        # Convertir resultados a elementos espaciales
        self.elements = []
        for bbox, text, confidence in ocr_results:
            if text.strip() and confidence > 0.1:
                # Convertir bbox de EasyOCR format a coordenadas
                if isinstance(bbox, list) and len(bbox) == 4:
                    # bbox es una lista de 4 puntos [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    bbox_rect = (int(x1), int(y1), int(x2), int(y2))
                else:
                    bbox_rect = bbox
                
                element = SpatialMathElement(text.strip(), bbox_rect, confidence)
                self.elements.append(element)
        
        if not self.elements:
            return ""
        
        # Ordenar elementos de izquierda a derecha, arriba a abajo
        self.elements.sort(key=lambda e: (e.center_y, e.center_x))
        
        # Detectar fracciones
        fractions = self._detect_fractions()
        
        # Detectar potencias/exponentes
        powers = self._detect_powers()
        
        # Detectar subíndices
        subscripts = self._detect_subscripts()
        
        # Detectar secuencias horizontales
        sequences = self._detect_sequences()
        
        # Construir texto final combinando todos los elementos
        result = self._build_final_text(fractions, powers, subscripts, sequences)
        
        if self.debug_mode:
            self._debug_print_analysis(fractions, powers, subscripts, sequences)
        
        return result
    
    def _detect_fractions(self) -> List[Dict]:
        """Detecta fracciones basándose en elementos verticalmente alineados"""
        fractions = []
        used_elements = set()
        
        for i, elem1 in enumerate(self.elements):
            if i in used_elements:
                continue
                
            # Buscar elementos debajo de este
            for j, elem2 in enumerate(self.elements):
                if j in used_elements or j <= i:
                    continue
                
                # Verificar si están verticalmente alineados y elem2 está debajo
                if self._are_vertically_aligned(elem1, elem2) and elem2.center_y > elem1.center_y:
                    
                    # Verificar si hay espacio vertical entre ellos (para la línea de fracción)
                    vertical_gap = elem2.y1 - elem1.y2
                    
                    # Debe haber un gap razonable (espacio para la línea)
                    if 5 <= vertical_gap <= 50:  # Ajustable según resolución
                        
                        # Verificar que no hay elementos en el medio horizontalmente
                        middle_y = (elem1.y2 + elem2.y1) // 2
                        has_middle_element = False
                        
                        for k, elem3 in enumerate(self.elements):
                            if k == i or k == j:
                                continue
                            # Si hay un elemento que cruza la línea de fracción
                            if (elem3.y1 <= middle_y <= elem3.y2 and 
                                self._overlaps_horizontally(elem1, elem3) or 
                                self._overlaps_horizontally(elem2, elem3)):
                                has_middle_element = True
                                break
                        
                        if not has_middle_element:
                            # Es una fracción válida
                            fractions.append({
                                'numerator': elem1,
                                'denominator': elem2,
                                'numerator_idx': i,
                                'denominator_idx': j,
                                'type': 'fraction'
                            })
                            used_elements.add(i)
                            used_elements.add(j)
                            break
        
        return fractions
    
    def _detect_powers(self) -> List[Dict]:
        """Detecta potencias basándose en elementos pequeños arriba a la derecha"""
        powers = []
        used_elements = set()
        
        for i, base_elem in enumerate(self.elements):
            if i in used_elements:
                continue
            
            # Buscar elementos candidatos a exponente
            for j, exp_elem in enumerate(self.elements):
                if j in used_elements or j == i:
                    continue
                
                # Verificar si es un candidato a exponente
                if self._is_valid_exponent(base_elem, exp_elem):
                    powers.append({
                        'base': base_elem,
                        'exponent': exp_elem,
                        'base_idx': i,
                        'exponent_idx': j,
                        'type': 'power'
                    })
                    used_elements.add(j)  # Solo marcar el exponente como usado
                    break
        
        return powers
    
    def _detect_subscripts(self) -> List[Dict]:
        """Detecta subíndices basándose en elementos pequeños abajo a la derecha"""
        subscripts = []
        used_elements = set()
        
        for i, base_elem in enumerate(self.elements):
            if i in used_elements:
                continue
            
            for j, sub_elem in enumerate(self.elements):
                if j in used_elements or j == i:
                    continue
                
                if self._is_valid_subscript(base_elem, sub_elem):
                    subscripts.append({
                        'base': base_elem,
                        'subscript': sub_elem,
                        'base_idx': i,
                        'subscript_idx': j,
                        'type': 'subscript'
                    })
                    used_elements.add(j)
                    break
        
        return subscripts
    
    def _detect_sequences(self) -> List[Dict]:
        """Detecta secuencias horizontales de elementos"""
        sequences = []
        
        # Agrupar elementos por línea horizontal
        lines = self._group_elements_by_line()
        
        for line in lines:
            if len(line) >= 3:  # Al menos 3 elementos para formar secuencia
                # Ordenar por posición horizontal
                line.sort(key=lambda e: e.center_x)
                
                # Detectar patrones de secuencia
                sequence_text = ' '.join([elem.text for elem in line])
                
                # Buscar patrones como "1 + 2 + 3" o "1 2 3"
                if self._is_sequence_pattern(sequence_text):
                    sequences.append({
                        'elements': line,
                        'text': sequence_text,
                        'type': 'sequence'
                    })
        
        return sequences
    
    def _are_vertically_aligned(self, elem1: SpatialMathElement, elem2: SpatialMathElement) -> bool:
        """Verifica si dos elementos están verticalmente alineados"""
        # Calcular overlap horizontal
        overlap_start = max(elem1.x1, elem2.x1)
        overlap_end = min(elem1.x2, elem2.x2)
        overlap_width = max(0, overlap_end - overlap_start)
        
        # Debe haber overlap significativo
        min_width = min(elem1.width, elem2.width)
        return overlap_width >= min_width * 0.3  # 30% de overlap mínimo
    
    def _overlaps_horizontally(self, elem1: SpatialMathElement, elem2: SpatialMathElement) -> bool:
        """Verifica si dos elementos se solapan horizontalmente"""
        return not (elem1.x2 < elem2.x1 or elem2.x2 < elem1.x1)
    
    def _is_valid_exponent(self, base: SpatialMathElement, exp: SpatialMathElement) -> bool:
        """Verifica si un elemento es un exponente válido del elemento base"""
        
        # 1. El exponente debe estar arriba y a la derecha del elemento base
        is_above = exp.center_y < base.center_y
        is_to_right = exp.center_x > base.center_x
        
        if not (is_above and is_to_right):
            return False
        
        # 2. El exponente debe ser notablemente más pequeño
        size_ratio = exp.area / base.area
        if size_ratio > 0.5:  # El exponente no debe ser más del 50% del tamaño de la base
            return False
        
        # 3. La distancia horizontal debe ser razonable
        horizontal_distance = exp.x1 - base.x2
        if horizontal_distance > base.width:  # No debe estar muy lejos
            return False
        
        # 4. La diferencia vertical debe ser apropiada
        vertical_offset = base.center_y - exp.center_y
        if vertical_offset > base.height:  # No debe estar muy arriba
            return False
        
        # 5. El exponente debe ser típicamente numérico o letra simple
        if len(exp.text) > 3:  # Exponentes suelen ser cortos
            return False
        
        return True
    
    def _is_valid_subscript(self, base: SpatialMathElement, sub: SpatialMathElement) -> bool:
        """Verifica si un elemento es un subíndice válido del elemento base"""
        
        # Similar a exponente pero abajo
        is_below = sub.center_y > base.center_y
        is_to_right = sub.center_x > base.center_x
        
        if not (is_below and is_to_right):
            return False
        
        # Más pequeño que la base
        size_ratio = sub.area / base.area
        if size_ratio > 0.5:
            return False
        
        # Distancia razonable
        horizontal_distance = sub.x1 - base.x2
        if horizontal_distance > base.width:
            return False
        
        vertical_offset = sub.center_y - base.center_y
        if vertical_offset > base.height:
            return False
        
        # Típicamente corto
        if len(sub.text) > 3:
            return False
        
        return True
    
    def _group_elements_by_line(self) -> List[List[SpatialMathElement]]:
        """Agrupa elementos que están en la misma línea horizontal"""
        lines = []
        used_elements = set()
        
        for i, elem in enumerate(self.elements):
            if i in used_elements:
                continue
            
            # Crear nueva línea con este elemento
            line = [elem]
            used_elements.add(i)
            
            # Buscar otros elementos en la misma línea
            for j, other_elem in enumerate(self.elements):
                if j in used_elements:
                    continue
                
                # Verificar si están en la misma línea horizontal
                if self._are_on_same_line(elem, other_elem):
                    line.append(other_elem)
                    used_elements.add(j)
            
            lines.append(line)
        
        return lines
    
    def _are_on_same_line(self, elem1: SpatialMathElement, elem2: SpatialMathElement) -> bool:
        """Verifica si dos elementos están en la misma línea horizontal"""
        # Calcular overlap vertical
        overlap_start = max(elem1.y1, elem2.y1)
        overlap_end = min(elem1.y2, elem2.y2)
        overlap_height = max(0, overlap_end - overlap_start)
        
        # Debe haber overlap vertical significativo
        min_height = min(elem1.height, elem2.height)
        return overlap_height >= min_height * 0.5  # 50% de overlap vertical
    
    def _is_sequence_pattern(self, text: str) -> bool:
        """Detecta si un texto representa una secuencia matemática"""
        # Patrones comunes de secuencias
        patterns = [
            r'\d+\s*[+\-*]\s*\d+\s*[+\-*]\s*\d+',  # 1 + 2 + 3
            r'[a-zA-Z]\d+\s*[+\-*]\s*[a-zA-Z]\d+',  # a1 + a2
            r'\d+\s+\d+\s+\d+',  # 1 2 3 (espacios)
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _build_final_text(self, fractions: List[Dict], powers: List[Dict], 
                         subscripts: List[Dict], sequences: List[Dict]) -> str:
        """Construye el texto final combinando todos los elementos detectados"""
        
        result_parts = []
        used_indices = set()
        
        # Procesar fracciones
        for fraction in fractions:
            num_text = fraction['numerator'].text
            den_text = fraction['denominator'].text
            result_parts.append(f"({num_text})/({den_text})")
            used_indices.add(fraction['numerator_idx'])
            used_indices.add(fraction['denominator_idx'])
        
        # Procesar potencias
        for power in powers:
            base_text = power['base'].text
            exp_text = power['exponent'].text
            
            # Usar símbolos unicode para exponentes comunes
            if exp_text == "2":
                result_parts.append(f"{base_text}²")
            elif exp_text == "3":
                result_parts.append(f"{base_text}³")
            elif exp_text == "4":
                result_parts.append(f"{base_text}⁴")
            elif exp_text == "5":
                result_parts.append(f"{base_text}⁵")
            else:
                result_parts.append(f"{base_text}^{{{exp_text}}}")
            
            used_indices.add(power['exponent_idx'])
            # No marcar la base como usada, puede ser parte de secuencia
        
        # Procesar subíndices
        for subscript in subscripts:
            base_text = subscript['base'].text
            sub_text = subscript['subscript'].text
            result_parts.append(f"{base_text}_{{{sub_text}}}")
            used_indices.add(subscript['subscript_idx'])
        
        # Procesar secuencias
        for sequence in sequences:
            seq_indices = [self.elements.index(elem) for elem in sequence['elements']]
            if not any(idx in used_indices for idx in seq_indices):
                result_parts.append(sequence['text'])
                used_indices.update(seq_indices)
        
        # Agregar elementos restantes
        for i, elem in enumerate(self.elements):
            if i not in used_indices:
                result_parts.append(elem.text)
        
        # Unir y limpiar resultado
        result = ' '.join(result_parts)
        
        # Aplicar correcciones finales
        result = self._apply_spatial_corrections(result)
        
        return result.strip()
    
    def _apply_spatial_corrections(self, text: str) -> str:
        """Aplica correcciones finales basadas en el análisis espacial"""
        corrected = text
        
        # Corregir espaciado alrededor de operadores
        import re
        corrected = re.sub(r'\s*([+\-*/=])\s*', r' \1 ', corrected)
        
        # Corregir fracciones mal formateadas
        corrected = re.sub(r'\(\s*([^)]+)\s*\)\s*/\s*\(\s*([^)]+)\s*\)', r'\1/\2', corrected)
        
        # Limpiar espacios múltiples
        corrected = re.sub(r'\s+', ' ', corrected)
        
        return corrected
    
    def _debug_print_analysis(self, fractions, powers, subscripts, sequences):
        """Imprime información de depuración"""
        print(f"\n🔍 ANÁLISIS ESPACIAL DEBUG:")
        print(f"📊 Elementos detectados: {len(self.elements)}")
        
        if fractions:
            print(f"📏 Fracciones encontradas: {len(fractions)}")
            for i, frac in enumerate(fractions):
                print(f"   {i+1}. {frac['numerator'].text}/{frac['denominator'].text}")
        
        if powers:
            print(f"⚡ Potencias encontradas: {len(powers)}")
            for i, pow in enumerate(powers):
                print(f"   {i+1}. {pow['base'].text}^{pow['exponent'].text}")
        
        if subscripts:
            print(f"₁ Subíndices encontrados: {len(subscripts)}")
            for i, sub in enumerate(subscripts):
                print(f"   {i+1}. {sub['base'].text}_{sub['subscript'].text}")
        
        if sequences:
            print(f"🔗 Secuencias encontradas: {len(sequences)}")
            for i, seq in enumerate(sequences):
                print(f"   {i+1}. {seq['text']}")

def enhanced_spatial_ocr_recognition(image_path: str) -> Dict[str, any]:
    """
    Función principal que combina OCR estándar con análisis espacial
    """
    try:
        # Importar EasyOCR
        import easyocr
        
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"No se pudo cargar la imagen: {image_path}"}
        
        # Inicializar EasyOCR
        reader = easyocr.Reader(['en'], gpu=False, verbose=False, download_enabled=False)
        
        # Realizar OCR con detalles de posición
        ocr_results = reader.readtext(image, detail=1)
        
        if not ocr_results:
            return {"text": "", "confidence": 0.0, "method": "spatial_no_results"}
        
        # Aplicar análisis espacial
        spatial_analyzer = SpatialMathAnalyzer()
        spatial_text = spatial_analyzer.analyze_spatial_math(ocr_results)
        
        # Calcular confianza promedio
        confidences = [conf for _, _, conf in ocr_results if conf > 0.1]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "text": spatial_text,
            "confidence": avg_confidence,
            "method": "spatial_analysis",
            "elements_detected": len(spatial_analyzer.elements),
            "raw_ocr_results": len(ocr_results)
        }
        
    except Exception as e:
        return {"error": f"Error en reconocimiento espacial: {str(e)}"}

if __name__ == "__main__":
    # Función de prueba
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = enhanced_spatial_ocr_recognition(image_path)
        
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"✅ Texto reconocido: '{result['text']}'")
            print(f"📊 Confianza: {result['confidence']:.2f}")
            print(f"🔧 Método: {result['method']}")
            print(f"📝 Elementos detectados: {result['elements_detected']}")
    else:
        print("Uso: python spatial_math_analyzer.py <ruta_imagen>")
