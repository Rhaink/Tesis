import os

import pandas as pd

# --- Configuración de Rutas ---
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"

# --- Archivos de Entrada ---
analysis_report_path = os.path.join(
    PROJECT_ROOT_DIR,
    "pulmones",
    "analysis",
    "dataset_morphology_analysis",
    "morphology_report_full_dataset.csv",
)
original_train_coords_path = os.path.join(
    PROJECT_ROOT_DIR, "coordenadas", "coordenadas_entrenamiento_1.csv"
)
original_test_coords_path = os.path.join(
    PROJECT_ROOT_DIR, "coordenadas", "coordenadas_prueba_1.csv"
)

# ==============================================================================
# === PARÁMETROS DE CURACIÓN ESTRICTA ===
# ==============================================================================
# Aquí podemos ajustar qué tan agresivos queremos ser.
# Por ejemplo, 0.15 significa eliminar el 15% más pequeño.
# 0.90 significa eliminar el 10% más grande (todo por encima del cuantil 90).

AREA_LOWER_QUANTILE = 0.15  # Antes era 0.10. Ahora eliminamos el 15% más pequeño.
ASPECT_LOWER_QUANTILE = 0.075  # Antes era 0.05.
ASPECT_UPPER_QUANTILE = 0.925  # Antes era 0.95. (Se calcula como 1 - 0.075)
SOLIDITY_LOWER_QUANTILE = 0.075  # Antes era 0.05.

# Nombres para los nuevos archivos de salida
output_suffix = f"_strict_curated_A{int(AREA_LOWER_QUANTILE * 100)}"
curated_train_coords_path = os.path.join(
    PROJECT_ROOT_DIR, "coordenadas", f"coordenadas_entrenamiento{output_suffix}.csv"
)
curated_test_coords_path = os.path.join(
    PROJECT_ROOT_DIR, "coordenadas", f"coordenadas_prueba{output_suffix}.csv"
)
curation_log_path = os.path.join(
    PROJECT_ROOT_DIR,
    "pulmones",
    "analysis",
    "dataset_morphology_analysis",
    f"curation_log{output_suffix}.txt",
)


def main():
    print(
        f"--- Proceso de Curación del Dataset con Criterios Estrictos: {output_suffix} ---"
    )

    # 1. Cargar el reporte de análisis morfológico
    print(f"Cargando reporte de análisis desde: {analysis_report_path}")
    try:
        df_report = pd.read_csv(analysis_report_path)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de reporte en {analysis_report_path}.")
        return

    # 2. Calcular los umbrales basados en los nuevos cuantiles
    area_threshold = df_report["area"].quantile(AREA_LOWER_QUANTILE)
    aspect_low_threshold = df_report["aspect_ratio"].quantile(ASPECT_LOWER_QUANTILE)
    aspect_high_threshold = df_report["aspect_ratio"].quantile(ASPECT_UPPER_QUANTILE)
    solidity_threshold = df_report["solidity"].quantile(SOLIDITY_LOWER_QUANTILE)

    with open(curation_log_path, "w") as log_file:
        log_file.write(f"--- Criterios de Curación Estricta ({output_suffix}) ---\n")
        log_file.write(
            f"1. Área < {area_threshold:.2f} (Cuantil {AREA_LOWER_QUANTILE})\n"
        )
        log_file.write(
            f"2. Relación de Aspecto < {aspect_low_threshold:.2f} (Cuantil {ASPECT_LOWER_QUANTILE})\n"
        )
        log_file.write(
            f"3. Relación de Aspecto > {aspect_high_threshold:.2f} (Cuantil {ASPECT_UPPER_QUANTILE})\n"
        )
        log_file.write(
            f"4. Solidez < {solidity_threshold:.2f} (Cuantil {SOLIDITY_LOWER_QUANTILE})\n\n"
        )

    # 3. Aplicar los criterios de exclusión
    condition_area = df_report["area"] < area_threshold
    condition_aspect = (df_report["aspect_ratio"] < aspect_low_threshold) | (
        df_report["aspect_ratio"] > aspect_high_threshold
    )
    condition_solidity = df_report["solidity"] < solidity_threshold

    to_remove = condition_area | condition_aspect | condition_solidity

    df_removed = df_report[to_remove]
    df_kept = df_report[~to_remove]
    good_image_names = set(df_kept["image_name"])

    print("\n--- Resumen de la Curación ---")
    print(f"Total de imágenes analizadas: {len(df_report)}")
    print(f"Imágenes a eliminar con criterios estrictos: {len(df_removed)}")
    print(f"Imágenes a mantener: {len(df_kept)}")

    with open(curation_log_path, "a") as log_file:
        log_file.write("--- Imágenes Eliminadas ---\n")
        df_removed.to_string(log_file)

    # 4. Filtrar y guardar los nuevos archivos de coordenadas
    print("\nProcesando y guardando nuevos archivos de coordenadas...")
    for original_path, curated_path, set_name in [
        (original_train_coords_path, curated_train_coords_path, "entrenamiento"),
        (original_test_coords_path, curated_test_coords_path, "prueba"),
    ]:
        df_orig = pd.read_csv(original_path)
        name_col = df_orig.columns[-1]
        df_curated = df_orig[df_orig[name_col].isin(good_image_names)]
        df_curated.to_csv(curated_path, index=False)
        print(
            f"  - Set {set_name}: Original={len(df_orig)}, Curado={len(df_curated)}. Guardado en: {os.path.basename(curated_path)}"
        )

    print("\n--- Proceso de Curación Finalizado ---")


if __name__ == "__main__":
    main()
