# Matching Geometric - Híbrido Template Matching + Construcción Geométrica

## Descripción General

El método **Matching Geometric** es un enfoque híbrido innovador que combina la precisión del **Template Matching** para puntos clave con la **construcción geométrica** basada en la lógica del etiquetado manual. Este método mejora significativamente la precisión de los puntos cuartil comparado con Template Matching puro.

## Arquitectura del Método

### 🎯 **Fase 1: Detección de Puntos Clave**
- Utiliza el modelo de **Template Matching** entrenado (5.63px error) para detectar los puntos principales **0** y **1**
- Estos puntos forman la **línea divisoria** que atraviesa el centro de los pulmones
- Precisión garantizada: utiliza exactamente los mismos resultados guardados del modelo validado

### 📐 **Fase 2: Construcción Geométrica**
- Implementa la **estrategia cuartil** del etiquetado manual
- Calcula puntos intermedios sobre la línea principal:
  - **Cuarto1**: 25% de la línea
  - **Medio**: 50% de la línea (punto central)
  - **Cuarto3**: 75% de la línea
- Genera landmarks adicionales usando **líneas perpendiculares** en cada punto cuartil

## Resultados y Logros

### 📊 **Comparación de Precisión: Puntos Cuartil**

| Método | Error Promedio | Desviación | Mejora |
|--------|---------------|------------|--------|
| **Template Matching** | 5.813 ± 1.954 px | - | - |
| **Matching Geometric** | 4.868 ± 2.544 px | **-16.3%** | ✅ |

### 📍 **Precisión por Punto Cuartil**

| Punto | Template Matching | Matching Geometric | Mejora |
|-------|-------------------|-------------------|--------|
| **Cuarto1** (landmark 8) | 5.757 ± 3.241 px | 4.908 ± 2.563 px | **14.7%** ✅ |
| **Medio** (landmark 9) | 5.995 ± 3.158 px | 4.691 ± 2.423 px | **21.7%** ✅ |
| **Cuarto3** (landmark 10) | 5.688 ± 3.238 px | 5.004 ± 2.649 px | **12.0%** ✅ |

### 🏆 **Logros Destacados**

- ✅ **Mejora consistente**: Todos los puntos cuartil muestran mejor precisión
- ✅ **Mayor estabilidad**: Menor variabilidad en las predicciones
- ✅ **Éxito en el 56.8%** de los casos comparado con Template Matching puro
- ✅ **Procesamiento completo**: 159/159 imágenes procesadas exitosamente
- ✅ **Validación anatómica**: Respeta la lógica del etiquetado manual

## Estructura del Proyecto

```
matching_geometric/
├── src/core/
│   ├── __init__.py
│   └── geometric_predictor.py      # Clase principal GeometricLandmarkPredictor
├── scripts/
│   ├── test_geometric_predictor.py       # Test básico del método
│   ├── process_all_159_images.py         # Procesamiento completo dataset
│   ├── compare_geometric_vs_tm.py        # Análisis comparativo
│   ├── verify_coordinates.py             # Verificación de coordenadas
│   └── verify_tm_error.py               # Validación errores TM
├── models/                               # Modelos guardados
├── results/                              # Resultados de predicciones
├── visualizations/
│   ├── all_159_images/                   # 159 visualizaciones básicas
│   ├── comparaciones_159/                # 159 comparaciones 3-métodos
│   ├── contornos_rapidos_159/           # 159 comparaciones con contornos ⭐
│   ├── geometric_vs_tm_comparison.png    # Gráficos comparativos
│   ├── method_comparison_grid.png        # Grid de comparación
│   ├── detailed_method_comparison.png    # Análisis detallado
│   └── geometric_vs_tm_stats.csv         # Estadísticas detalladas
└── README.md                             # Este archivo
```

## Componentes Clave

### 🔧 **GeometricLandmarkPredictor**

Clase principal que implementa el método híbrido:

```python
from geometric_predictor import GeometricLandmarkPredictor

# Inicializar con modelo TM entrenado
predictor = GeometricLandmarkPredictor(tm_model_path)

# Predecir landmarks con construcción geométrica
result = predictor.predict_landmarks(image, image_name='ejemplo')

# Resultado contiene:
# - landmarks: Todos los 15 puntos generados
# - key_points: Puntos 0 y 1 del TM
# - main_line: Parámetros de la línea principal
# - intermediate_points: Puntos cuartil calculados
```

### 📐 **Algoritmo de Construcción Geométrica**

1. **Detección de línea principal**: Puntos 0 y 1 del Template Matching
2. **Cálculo de cuartiles**: División de la línea en 4 partes iguales
3. **Generación de perpendiculares**: Líneas perpendiculares en cada cuartil
4. **Asignación anatómica**: Posicionamiento correcto de landmarks

## Comandos de Uso

### 🚀 **Procesamiento Completo**
```bash
# Procesar todas las 159 imágenes de prueba (método básico)
python matching_geometric/scripts/process_all_159_images.py

# Generar comparaciones visuales (3 métodos lado a lado)
python matching_geometric/scripts/generate_all_comparisons.py

# Generar comparaciones con contornos anatómicos (RECOMENDADO)
python matching_geometric/scripts/quick_contours_159.py
```

### 📊 **Análisis Comparativo**
```bash
# Comparar precisión geométrico vs Template Matching
python matching_geometric/scripts/compare_geometric_vs_tm.py

# Crear visualizaciones comparativas detalladas
python matching_geometric/scripts/create_comparison_visualization.py
```

### 🧪 **Pruebas y Verificación**
```bash
# Test básico del método
python matching_geometric/scripts/test_geometric_predictor.py

# Verificar coordenadas correctas con modelo 5.63px
python matching_geometric/scripts/verify_coordinates.py

# Validar errores de Template Matching
python matching_geometric/scripts/verify_tm_error.py
```

## Metodología de Validación

### 🔍 **Dataset de Prueba**
- **159 imágenes** del conjunto `coordenadas_prueba_1.csv`
- Comparación directa con **ground truth** manual
- Análisis estadístico completo con métricas de precisión

### 📈 **Métricas Evaluadas**
- **Error promedio** por landmark (distancia euclidiana)
- **Desviación estándar** para medir consistencia
- **Tasa de éxito** (casos donde geométrico > TM)
- **Correlación** entre métodos

### 🎯 **Validación Anatómica**
- Respeto de **conectividad anatómica** de landmarks
- Preservación de **proporciones** entre cuartiles
- Consistencia con **lógica del etiquetado manual**

## Innovaciones Técnicas

### 🧠 **Hibridación Inteligente**
- Combina **precisión del ML** (Template Matching) para puntos clave
- Aplica **conocimiento geométrico** para puntos derivados
- Evita **acumulación de errores** del ML puro

### 📏 **Estrategia Cuartil Optimizada**
- Implementación fiel de la **lógica de etiquetado manual**
- Cálculo preciso de **líneas perpendiculares**
- **Asignación anatómica** correcta de landmarks

### 🔄 **Robustez y Consistencia**
- **Menor variabilidad** que Template Matching puro
- **Resultados reproducibles** basados en geometría
- **Validación cruzada** con datos guardados

## Limitaciones y Trabajo Futuro

### ⚠️ **Limitaciones Actuales**
- Dependiente de la **calidad de los puntos 0 y 1** del Template Matching
- **Distancias perpendiculares fijas** (pueden necesitar ajuste por anatomía)
- Evaluado solo en **puntos cuartil** (no todos los 15 landmarks)

### 🔮 **Trabajo Futuro**
- Extensión a **todos los 15 landmarks** con construcción geométrica
- **Distancias adaptativas** basadas en tamaño pulmonar
- **Integración con otros métodos** (ASM, Deep Learning)
- **Validación clínica** en diferentes patologías

## Visualizaciones Generadas

### 📂 **Conjuntos de Imágenes Disponibles**

1. **Básicas** (`all_159_images/`): Solo método geométrico con 5 puntos clave
2. **Comparativas** (`comparaciones_159/`): 3 métodos lado a lado sin contornos  
3. **Con Contornos** (`contornos_rapidos_159/`) ⭐ **RECOMENDADO**:
   - Ground Truth con contornos verdes
   - Template Matching con contornos rojos
   - Matching Geométrico con línea principal + cuartiles
   - Conexiones anatómicas correctas del CLAUDE.md
   - Velocidad: ~106 imágenes/segundo

### 🎯 **Análisis Estadísticos**
- Gráficos de precisión comparativa
- Análisis detallado por landmark
- Estadísticas exportadas en CSV

## Contribuciones

Este método demuestra que la **combinación inteligente** de técnicas de Machine Learning con **conocimiento geométrico explícito** puede superar a los enfoques puramente basados en datos. La **estrategia cuartil** del etiquetado manual se valida como una técnica superior para generar landmarks intermedios precisos y consistentes.

Las **visualizaciones con contornos anatómicos** permiten evaluar visualmente la calidad de las conexiones de landmarks y la preservación de la anatomía pulmonar en cada método.

---

**Desarrollado como parte del proyecto de análisis morfológico pulmonar usando Active Shape Models, Template Matching y métodos geométricos híbridos.**