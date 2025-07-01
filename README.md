# Tesis: Análisis de Morfología Pulmonar y Detección de COVID-19

Este proyecto es una colección de módulos y scripts desarrollados para el análisis de la morfología pulmonar y la detección de COVID-19 a partir de imágenes de radiografías de tórax. Utiliza técnicas de Modelos de Forma Activa (ASM) y Redes Neuronales Convolucionales (CNN) para el procesamiento y análisis de imágenes médicas.

## Estructura del Proyecto

El repositorio está organizado en varias carpetas principales, cada una con un propósito específico:

*   **`pulmones/`**: Contiene la implementación principal de un sistema ASM completo. Incluye el código fuente para los modelos de forma (`ShapeModel`) y apariencia (`AppearanceModel`, `MultiLevelAppearanceModel`), así como scripts para el entrenamiento de estos modelos y el análisis de los resultados.
    *   `pulmones/scripts/`: Scripts ejecutables para tareas como el análisis de morfología del dataset, curación de datos, procesamiento y deformación de imágenes, y entrenamiento de ASM con aumento de datos.
    *   `pulmones/src/core/`: Implementaciones de las clases principales de ASM (modelos de forma, apariencia y el ajustador ASM).
    *   `pulmones/src/utils/`: Funciones de utilidad para ASM, incluyendo carga de landmarks, alineación Procrustes, PCA, procesamiento de imágenes y visualización.
    *   `pulmones/data/`: Contiene datos de entrada, como coordenadas de prueba e imágenes.
    *   `pulmones/models/`: Almacena los modelos ASM entrenados (forma y apariencia).
    *   `pulmones/results/`: Guarda los resultados de las evaluaciones y visualizaciones del modelo ASM.

*   **`template_matching/`**: Implementación de Template Matching con PCA/Eigenpatches como método alternativo de detección de landmarks. Logra un error promedio de **5.63 ± 1.03 pixeles** en 159 imágenes de prueba.
    *   `template_matching/src/core/`: Implementaciones del modelo de eigenpatches y predictor de landmarks multi-escala.
    *   `template_matching/scripts/`: Scripts para entrenamiento, procesamiento e visualización interactiva.
    *   `template_matching/models/`: Modelos entrenados de Template Matching.
    *   `template_matching/results/`: Resultados de predicción y evaluación (`results_coordenadas_prueba_1.pkl`).
    *   `template_matching/visualizations/`: Visualizaciones generadas para todas las imágenes de prueba.

*   **`delaunay_morphing/`**: **NUEVO** - Implementación de morfología pulmonar usando triangulación de Delaunay. Combina la precisión del Template Matching con técnicas de morfología similares a ASM.
    *   `delaunay_morphing/src/core/`: Motor de morfología Delaunay (`DelaunayLungMorpher`) con conectividad anatómica correcta.
    *   `delaunay_morphing/processed_159/`: Datos procesados de las 159 imágenes y forma canónica calculada.
    *   `delaunay_morphing/correct_tm_visualizations/`: **159 visualizaciones completas** mostrando landmarks exactos de Template Matching, morfología hacia forma canónica y triangulación de Delaunay.
    *   Scripts de procesamiento: `process_all_159_images.py`, `fix_correct_tm_order.py`, `complete_remaining_correct.py`.

*   **`coordenadas/`**: Directorio que almacena varios archivos CSV con coordenadas de puntos de referencia (landmarks) para diferentes conjuntos de datos (entrenamiento, prueba, originales, alineados, etc.). Estos archivos son fundamentales para el entrenamiento y la evaluación de los modelos de forma.

*   **`indices/`**: Contiene archivos CSV con índices o metadatos que asocian las imágenes con sus respectivas coordenadas y categorías.

*   **`COVID-19_Radiography_Dataset/`**: Este directorio contiene el conjunto de datos de imágenes de radiografías de tórax utilizado en el proyecto. Está organizado por categorías (COVID, Lung Opacity, Normal, Viral Pneumonia) y contiene subcarpetas para imágenes y máscaras. **Nota:** Esta carpeta no está incluida en el control de versiones de Git debido a su tamaño.

## Instalación

Para configurar el entorno y ejecutar los scripts, sigue estos pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd Tesis
    ```
2.  **Instalar dependencias:**
    Asegúrate de tener `pip` instalado. Luego, instala las bibliotecas necesarias:
    ```bash
    pip install -r requirements.txt
    ```
    Se recomienda usar un entorno virtual (e.g., `venv` o `conda`) para gestionar las dependencias.

## Uso

El proyecto contiene varios scripts principales para el entrenamiento y la evaluación de los modelos. A continuación, se describen algunos de los más relevantes:

*   **Entrenamiento de Modelos ASM:**
    *   `pulmones/scripts/train_full_augmentation_asm.py`: Entrena un modelo ASM completo utilizando aumento de datos de forma y apariencia.

*   **Template Matching (Precisión: 5.63±1.03px):**
    *   `template_matching/scripts/train_eigenpatches.py`: Entrena el modelo de Template Matching con eigenpatches.
    *   `template_matching/scripts/process_all_images.py`: Procesa todas las 159 imágenes de prueba.
    *   `template_matching/scripts/interactive_viewer.py`: Visualizador interactivo de resultados reales.

*   **Delaunay Morphing (NUEVO):**
    *   `delaunay_morphing/process_all_159_images.py`: Procesa las 159 imágenes con morfología Delaunay usando landmarks de Template Matching.
    *   `delaunay_morphing/fix_correct_tm_order.py`: Genera las 159 visualizaciones correctas con landmarks exactos de TM.

*   **Evaluación y Análisis:**
    *   `pulmones/scripts/analyze_dataset_morphology.py`: Realiza un análisis morfológico del dataset.
    *   `pulmones/scripts/process_and_warp_all.py`: Procesa y deforma imágenes utilizando los modelos entrenados.

Para ejecutar un script, navega a la carpeta raíz del proyecto (`Tesis/`) y usa el comando `python`:

```bash
# Activar entorno virtual
source pulmones/.venv/bin/activate

# Entrenar ASM
python pulmones/scripts/train_full_augmentation_asm.py

# Entrenar Template Matching
python template_matching/scripts/train_eigenpatches.py

# Generar visualizaciones Delaunay (requiere Template Matching entrenado)
python delaunay_morphing/fix_correct_tm_order.py
```

## Resultados y Modelos

Los modelos entrenados se guardan en las respectivas carpetas `models/` dentro de cada subproyecto:

*   **`pulmones/models/`**: Modelos ASM entrenados (forma y apariencia).
*   **`template_matching/models/`**: Modelos de Template Matching entrenados con eigenpatches.
    *   `landmark_predictor_*.pkl`: Predictor multi-escala entrenado.
*   **`template_matching/results/`**: 
    *   `results_coordenadas_prueba_1.pkl`: **Archivo clave** - Contiene las predicciones exactas de las 159 imágenes de prueba con error de 5.63±1.03px.

### Visualizaciones Generadas

*   **`template_matching/visualizations/all_test_images/`**: Visualizaciones de Template Matching para las 159 imágenes.
*   **`delaunay_morphing/correct_tm_visualizations/`**: **159 visualizaciones completas** de morfología Delaunay con:
    *   Landmarks exactos de Template Matching (error 5.63±1.03px)
    *   Morfología hacia forma canónica usando triangulación de Delaunay  
    *   Conectividad anatómica correcta de landmarks pulmonares
    *   Triangulación superpuesta con métricas de calidad

### Datos Procesados

*   **`delaunay_morphing/processed_159/`**: 
    *   `canonical_shape.npy`: Forma canónica calculada a partir de las 159 predicciones de Template Matching
    *   Datos de morfología procesados para reproducir resultados

Los resultados de las evaluaciones, visualizaciones y análisis se encuentran en las carpetas `results/` y `analysis/` de cada subproyecto. Estos incluyen gráficos, informes CSV y visualizaciones de las deformaciones.
