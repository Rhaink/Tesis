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

*   **`pulmones_1d/`**: Módulo relacionado con ASM, posiblemente enfocado en modelos de apariencia 1D o una implementación simplificada del pipeline ASM. Sigue una estructura similar a `pulmones/`.

*   **`pulmones_2d/`**: Módulo ASM que explora modelos de apariencia 2D o implementaciones específicas para características 2D. También replica la estructura de `pulmones/`.

*   **`pulmones_asm_delunay_morphing/`**: Este módulo parece investigar técnicas de morphing basadas en la triangulación de Delaunay, aplicadas en el contexto de los Modelos de Forma Activa.

*   **`pulmones_cnn/`**: Contiene la implementación de modelos de Redes Neuronales Convolucionales (CNN) para tareas como la regresión de puntos de referencia o la clasificación de imágenes.
    *   `pulmones_cnn/scripts/`: Scripts para la preparación de datos para CNN, entrenamiento de modelos CNN y evaluación.
    *   `pulmones_cnn/src/core/`: Incluye la definición del modelo CNN (`cnn_model.py`).
    *   `pulmones_cnn/cnn_data_augmented/`: Almacena datos aumentados específicos para el entrenamiento de CNN.
    *   `pulmones_cnn/models/`: Guarda los modelos CNN entrenados.
    *   `pulmones_cnn/results/`: Contiene los resultados de las evaluaciones de los modelos CNN.

*   **`pulmones_versionantigua_morfing/`**: Una versión anterior o alternativa del módulo de morphing, o una implementación con diferentes enfoques y resultados.

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
    *   `pulmones_1d/scripts/train_asm.py`: Entrena un modelo ASM para la versión 1D.
    *   `pulmones_2d/scripts/train_asm.py`: Entrena un modelo ASM para la versión 2D.

*   **Entrenamiento de Modelos CNN:**
    *   `pulmones_cnn/scripts/prepare_cnn_data.py`: Prepara los datos para el entrenamiento del modelo CNN.
    *   `pulmones_cnn/scripts/train_cnn.py`: Entrena el modelo CNN para la regresión de landmarks.

*   **Evaluación y Análisis:**
    *   `pulmones/scripts/analyze_dataset_morphology.py`: Realiza un análisis morfológico del dataset.
    *   `pulmones/scripts/process_and_warp_all.py`: Procesa y deforma imágenes utilizando los modelos entrenados.
    *   `pulmones_1d/scripts/evaluate_asm.py`: Evalúa el rendimiento del modelo ASM 1D.
    *   `pulmones_2d/scripts/evaluate_asm.py`: Evalúa el rendimiento del modelo ASM 2D.
    *   `pulmones_cnn/scripts/evaluate_asm.py`: Evalúa el rendimiento del modelo CNN.

Para ejecutar un script, navega a la carpeta raíz del proyecto (`Tesis/`) y usa el comando `python`:

```bash
python pulmones/scripts/train_full_augmentation_asm.py
```

## Resultados y Modelos

Los modelos entrenados se guardan en las respectivas carpetas `models/` dentro de cada subproyecto (e.g., `pulmones/models/`, `pulmones_cnn/models/`).

Los resultados de las evaluaciones, visualizaciones y análisis se encuentran en las carpetas `results/` y `analysis/` de cada subproyecto (e.g., `pulmones/results/`, `pulmones_cnn/analysis/`). Estos incluyen gráficos, informes CSV y visualizaciones de las deformaciones.
