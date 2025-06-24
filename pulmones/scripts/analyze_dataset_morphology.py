import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuración de Rutas ---
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")

if SRC_DIR_PULMONES not in sys.path:
    sys.path.append(SRC_DIR_PULMONES)

from utils import asm_utils

# ==============================================================================
# === FUNCIONES DE CÁLCULO DE MÉTRICAS ===
# ==============================================================================


def calculate_shape_metrics(shape_coords, contour_indices):
    """
    Calcula métricas morfológicas para una forma dada.
    """
    if shape_coords.size == 0 or np.any(np.isnan(shape_coords)):
        return {"area": 0, "aspect_ratio": 0, "solidity": 0, "width": 0, "height": 0}

    # Bounding Box
    min_x, min_y = np.min(shape_coords, axis=0)
    max_x, max_y = np.max(shape_coords, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = width / height if height > 0 else 0

    # Métricas basadas en el contorno
    contour_points = shape_coords[contour_indices].astype(np.int32)
    area = cv2.contourArea(contour_points)

    if area > 0:
        hull = cv2.convexHull(contour_points)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
    else:
        solidity = 0

    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "width": width,
        "height": height,
    }


# ==============================================================================
# === SCRIPT PRINCIPAL ===
# ==============================================================================


def main():
    print("--- Análisis Morfológico del Dataset Completo ---")

    # --- Parámetros ---
    # Analizaremos ambos datasets para entender la distribución completa
    training_coords_csv = os.path.join(
        PROJECT_ROOT_DIR, "coordenadas", "coordenadas_entrenamiento_1.csv"
    )
    test_coords_csv = os.path.join(
        PROJECT_ROOT_DIR, "coordenadas", "coordenadas_prueba_1.csv"
    )

    analysis_output_dir = os.path.join(
        PROJECT_ROOT_DIR, "pulmones", "analysis", "dataset_morphology_analysis"
    )
    report_csv_path = os.path.join(
        analysis_output_dir, "morphology_report_full_dataset.csv"
    )

    num_landmarks = 15
    # Índices que forman el contorno exterior para calcular área y solidez
    contour_indices = [0, 12, 3, 5, 7, 14, 1, 13, 6, 4, 2, 11]

    os.makedirs(analysis_output_dir, exist_ok=True)

    # 1. Cargar ambos conjuntos de datos
    print("Cargando datos de entrenamiento y prueba...")
    train_shapes, train_names = asm_utils.load_landmarks(
        training_coords_csv, num_landmarks
    )
    test_shapes, test_names = asm_utils.load_landmarks(test_coords_csv, num_landmarks)

    all_shapes = np.concatenate([train_shapes, test_shapes], axis=0)
    all_names = train_names + test_names

    # Añadir una columna para saber si es de entrenamiento o prueba
    set_types = ["train"] * len(train_names) + ["test"] * len(test_names)

    print(f"Total de formas a analizar: {len(all_shapes)}")

    # 2. Calcular métricas para cada forma
    all_metrics = []
    print("Calculando métricas morfológicas para cada forma...")
    for i in tqdm(range(len(all_shapes)), desc="Procesando formas"):
        shape = all_shapes[i]
        metrics = calculate_shape_metrics(shape, contour_indices)
        metrics["image_name"] = all_names[i]
        metrics["dataset_type"] = set_types[i]
        all_metrics.append(metrics)

    # 3. Guardar el reporte detallado
    df_report = pd.DataFrame(all_metrics)
    df_report.to_csv(report_csv_path, index=False)
    print(f"\nReporte morfológico detallado guardado en: {report_csv_path}")

    # 4. Generar visualizaciones de las distribuciones
    print("Generando visualizaciones de las distribuciones de métricas...")

    # Histograma de Relación de Aspecto
    plt.figure(figsize=(10, 6))
    plt.hist(df_report["aspect_ratio"], bins=50, alpha=0.7)
    plt.title("Distribución de la Relación de Aspecto (Ancho/Alto) del Tórax")
    plt.xlabel("Relación de Aspecto")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(analysis_output_dir, "dist_aspect_ratio.png"))
    plt.close()

    # Histograma de Área
    plt.figure(figsize=(10, 6))
    plt.hist(df_report["area"], bins=50, alpha=0.7, color="green")
    plt.title("Distribución del Área del Polígono Pulmonar")
    plt.xlabel("Área (píxeles^2)")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(analysis_output_dir, "dist_area.png"))
    plt.close()

    # Gráfico de Dispersión: Área vs Relación de Aspecto
    plt.figure(figsize=(12, 8))
    plt.scatter(df_report["area"], df_report["aspect_ratio"], alpha=0.5, s=20)
    plt.title("Área vs. Relación de Aspecto")
    plt.xlabel("Área del Polígono")
    plt.ylabel("Relación de Aspecto (Ancho/Alto)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(analysis_output_dir, "scatter_area_vs_aspect_ratio.png"))
    plt.close()

    print(f"Visualizaciones guardadas en: {analysis_output_dir}")

    # 5. Imprimir resumen y sugerencias
    print("\n--- Resumen Estadístico Morfológico ---")
    print(df_report[["area", "aspect_ratio", "solidity", "width", "height"]].describe())

    print("\n--- Posibles Outliers Morfológicos (Sugerencias para inspección) ---")
    # Sugerir inspeccionar los extremos de las distribuciones
    aspect_ratio_q95 = df_report["aspect_ratio"].quantile(0.95)
    area_q05 = df_report["area"].quantile(0.05)

    print("\nSe sugiere inspeccionar imágenes con:")
    print(
        f"  - Relación de aspecto > {aspect_ratio_q95:.2f} (el 5% más ancho/aplastado)"
    )
    print(f"  - Área < {area_q05:.2f} (el 5% más pequeño)")

    # Encontrar y listar algunos ejemplos
    outliers_aspect_ratio = df_report[df_report["aspect_ratio"] > aspect_ratio_q95]
    outliers_area = df_report[df_report["area"] < area_q05]

    print(
        "\nEjemplos de outliers por Relación de Aspecto (posiblemente pediátricos/inusuales):"
    )
    print(outliers_aspect_ratio[["image_name", "aspect_ratio"]].head())

    print("\nEjemplos de outliers por Área (posiblemente pediátricos/mal anotados):")
    print(outliers_area[["image_name", "area"]].head())

    print("\n--- Análisis Morfológico Completado ---")


if __name__ == "__main__":
    main()
