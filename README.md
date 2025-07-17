# Escáner OCR con Motor de Demostraciones Matemáticas

Una aplicación avanzada de interfaz gráfica desarrollada en Python con PyQt5 que escanea texto matemático manuscrito y **genera demostraciones matemáticas formales** usando métodos como Gentzen y inducción matemática.

## 🚀 Características Principales

### **Reconocimiento OCR Avanzado** 🔍
- **Sistema Multi-Estrategia**: 6 variantes de preprocesamiento × 7 configuraciones OCR = 42 intentos de reconocimiento
- **EasyOCR optimizado** para manuscritos matemáticos con análisis espacial
- **Análisis Espacial**: Detección inteligente de fracciones manuscritas (números apilados verticalmente) y potencias (números pequeños arriba-derecha)
- **Corrección automática** de 50+ símbolos matemáticos mal reconocidos
- **Preprocesamiento especializado** para elementos complejos (raíces, sumatorias, secuencias con "...")

### **Motor de Demostraciones Matemáticas** 🧮
- **Inducción Matemática Universal**: Sistema inteligente que detecta y demuestra automáticamente patrones
- **Método de Gentzen**: Cálculo de secuentes para lógica proposicional
- **Teoría de Conjuntos**: Demostraciones de relaciones de subconjuntos, unión, intersección
- **Análisis Automático**: Detección inteligente del tipo de problema matemático
- **Generación de Pruebas Formales**: LaTeX con formato matemático profesional

### **Reconocimiento Espacial de Elementos Manuscritos** 📐
- **Fracciones Manuscritas**: Reconoce "dos números uno encima del otro separados por línea" usando análisis de posición vertical
- **Potencias Contextuales**: Distingue entre número normal y exponente basándose en tamaño y posición (pequeño, arriba-derecha)
- **Subíndices**: Detección de números pequeños abajo-derecha para notación científica
- **Secuencias**: Reconocimiento inteligente de patrones como "1+2+3+...+n"

### **Conversión Automática a LaTeX**
- **Ecuaciones matemáticas** con formato profesional usando paquete bussproofs
- **Estructuras de demostración** completas con numeración automática
- **Símbolos matemáticos** especializados para todo tipo de notación
- **Documentos LaTeX compilables** listos para publicación académica

## Requisitos del Sistema

### Software Requerido
- **Python 3.7 o superior**: https://python.org

### Dependencias Python (se instalan automáticamente)

**Principales:**
- PyQt5==5.15.10 (Interfaz gráfica)
- easyocr==1.7.0 (OCR optimizado para manuscritos con análisis espacial)
- Pillow==10.0.1 (Procesamiento de imágenes)
- opencv-python==4.8.1.78 (Visión computacional)
- numpy==1.24.3 (Computación numérica)

**Matemáticas y Demostraciones:**
- sympy==1.12 (Matemática simbólica)
- torch==2.0.1 (PyTorch para EasyOCR)

## Instalación Rápida

### Windows
```batch
# Ejecutar en CMD o PowerShell
git clone <repositorio>
cd Escaner_de_demostaciones
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### Linux/macOS
```bash
git clone <repositorio>
cd Escaner_de_demostaciones
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
python3 main.py
```

## Uso

### Flujo de Trabajo Simplificado

1. **Ejecutar**: `python main.py`
2. **Cargar imagen**: Botón "📁 Cargar Imagen"
3. **Procesar**: Botón "🔍 Procesar con OCR" (usa automáticamente el sistema multi-estrategia)
4. **Ver resultados**: La aplicación muestra automáticamente:
   - Texto extraído con corrección de símbolos matemáticos
   - Análisis del tipo de problema detectado
   - Demostración matemática generada
   - Código LaTeX compilable

5. **Guardar**: Botón "💾 Guardar Resultados" para exportar todos los archivos

### Tipos de Problemas que Reconoce Automáticamente

#### **Inducción Matemática**
```
Ejemplos que detecta:
- "Demostrar por inducción que 1+2+3+...+n = n(n+1)/2"
- "Probar que 1²+2²+3²+...+n² = n(n+1)(2n+1)/6"
- "Para todo n ∈ ℕ, 2^n > n"
- "1+3+5+...+(2n-1) = n²"
```

#### **Teoría de Conjuntos**
```
Problemas que resuelve:
- "Demostrar que B ⊇ A ∪ B"
- "Probar que A ∩ B ⊆ A"
- "Si A ⊆ B y B ⊆ C, entonces A ⊆ C"
```

#### **Lógica Proposicional (Gentzen)**
```
Reconoce patrones como:
- "P → Q, P ⊢ Q" (Modus Ponens)
- "P ∧ Q ⊢ P" (Eliminación de conjunción)
- "P ⊢ P ∨ Q" (Introducción de disyunción)
```

## Estructura del Proyecto

```
Escaner_de_demostaciones/
├── main.py                     # Aplicación principal GUI
├── proof_engine.py             # Motor de demostraciones matemáticas
├── enhanced_ocr_system.py      # Sistema OCR multi-estrategia avanzado
├── spatial_math_analyzer.py    # Análisis espacial para manuscritos
├── ocr_config.py              # Configuraciones OCR especializadas
├── math_ocr_optimizer.py      # Optimizaciones para símbolos matemáticos
├── set_theory_ocr.py          # Especialización en teoría de conjuntos
├── run.py                     # Launcher alternativo
├── instalar_y_configurar.bat  # Instalador Windows (Para instalar dependencias, no es un instalador .exe)
├── requirements.txt           # Dependencias Python
├── README.md                  # Esta documentación
└── OCR_MEJORADO_GUIA.md      # Guía técnica del sistema OCR (OCR es el sistema de lectura manuscrita y Reescritura a computadora)
```

## Archivos Principales

### **main.py**
- Interfaz gráfica principal con PyQt5
- Integración de todos los sistemas OCR y demostración
- Manejo de archivos y exportación
- Sistema de pestañas para diferentes resultados

### **proof_engine.py**
- `ProofAssistant`: Coordinador principal de demostraciones
- `InductionProofSystem`: Sistema de inducción matemática universal
- `GentzenProofSystem`: Cálculo de secuentes para lógica
- Análisis automático de patrones matemáticos
- Generación de LaTeX con formato profesional

### **enhanced_ocr_system.py**
- `EnhancedOCREngine`: Sistema OCR con 42 estrategias combinadas
- `AdvancedImagePreprocessor`: 6 variantes de preprocesamiento
- Integración con análisis espacial para manuscritos
- Corrección automática de 50+ símbolos matemáticos
- Priorización de resultados por confianza

### **spatial_math_analyzer.py**
- `SpatialMathAnalyzer`: Análisis posicional de elementos manuscritos
- Detección de fracciones por alineación vertical
- Reconocimiento de potencias por tamaño y posición
- Análisis de subíndices y secuencias matemáticas

## Características Técnicas Avanzadas

### **Sistema Multi-Estrategia de OCR**
- **6 Variantes de Preprocesamiento**: Original, escala de grises, binario, filtrado, enfoque, contraste mejorado
- **7 Configuraciones OCR**: Diferentes configuraciones de EasyOCR optimizadas para manuscritos
- **Priorización Inteligente**: Análisis espacial tiene prioridad sobre OCR tradicional
- **Validación Cruzada**: Comparación entre múltiples resultados para mayor precisión

### **Reconocimiento Espacial Avanzado**
- **Detección de Fracciones**: Análisis de posición vertical y gaps horizontales para identificar numerador/denominador
- **Reconocimiento de Potencias**: Criterios de tamaño (0.3-0.8 del texto base) y posición (arriba-derecha)
- **Análisis de Subíndices**: Detección de elementos pequeños abajo-derecha
- **Secuencias Matemáticas**: Reconocimiento de patrones "1+2+3+...+n" por análisis contextual

### **Motor de Inducción Universal**
- **Detección Automática de Patrones**: Reconoce automáticamente el tipo de serie matemática
- **Generación de Demostraciones**: Produce pruebas completas con caso base, hipótesis inductiva y paso inductivo
- **LaTeX Profesional**: Genera documentos compilables con paquete bussproofs
- **Validación Matemática**: Verifica la corrección de las fórmulas detectadas

## Resolución de Problemas

### **OCR no reconoce bien los manuscritos**
- El sistema usa automáticamente 42 estrategias diferentes
- El análisis espacial maneja fracciones y potencias manuscritas
- Para mejor resultado: usar imágenes de alta resolución y buen contraste

### **No detecta el tipo de demostración**
- El sistema analiza automáticamente el texto extraído
- Verifica que la imagen contenga palabras clave como "demostrar", "inducción", "para todo n"
- Revisa el resultado en la pestaña "Análisis" para ver la confianza de detección

### **Error de memoria con EasyOCR**
- Reduce el tamaño de la imagen antes de procesar
- Cierra otras aplicaciones que consuman mucha memoria
- El sistema optimiza automáticamente el uso de memoria

## Desarrollo y Contribuciones

### **Agregar Nuevos Patrones de Inducción**
Editar `proof_engine.py` en el método `_analyze_mathematical_formula`:
```python
if "nuevo_patron" in clean_text:
    return {"series_type": "NUEVO_TIPO", "confidence": 0.9}
```

### **Mejorar Reconocimiento Espacial**
Modificar `spatial_math_analyzer.py` para agregar nuevos tipos de elementos:
```python
def _detect_nuevo_elemento(self, results):
    # Implementar lógica de detección
    pass
```

### **Personalizar Corrección de Símbolos**
Actualizar el diccionario en `enhanced_ocr_system.py`:
```python
self.symbol_corrections = {
    'simbolo_incorrecto': 'símbolo_correcto',
    # ...
}
```
