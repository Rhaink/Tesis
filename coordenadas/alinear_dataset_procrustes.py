# alinear_dataset_procrustes.py

import os
import pandas as pd
import numpy as np
import cv2
import math
import shutil
from scipy.linalg import svd # Para encontrar la rotación óptima

# --- Configuración ---
# Ajusta estas rutas según tu estructura de directorios
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Asume que el script está en la carpeta base del proyecto
# base_dir = "/ruta/a/tu/proyecto/" # O define la ruta absoluta

# Archivo CSV de entrada con coordenadas (para 64x64)
coords_file = os.path.join(base_dir, "coordenadas/coordenadas_maestro_1.csv")

# Directorio con las imágenes redimensionadas a 64x64
# IMPORTANTE: Asegúrate de que este directorio exista y contenga las imágenes
# nombradas como en la última columna del CSV.
image_dir_64x64 = os.path.join(base_dir, "dataset/dataset_maestro_1")

# Directorio de salida para imágenes alineadas
output_image_dir = os.path.join(base_dir, "dataset/dataset_procrustes_maestro_1")
# Directorio de salida para el nuevo archivo de coordenadas
output_coords_file = os.path.join(base_dir, "coordenadas/coordenadas_procrustes_maestro_1.csv")

IMG_SIZE = 64

# --- Crear directorios de salida ---
os.makedirs(output_image_dir, exist_ok=True)

# --- Funciones Auxiliares ---

def load_shapes_from_csv(csv_path):
    """Carga las coordenadas y nombres de archivo desde el CSV."""
    try:
        df = pd.read_csv(csv_path, header=None)
        num_cols = len(df.columns)
        if num_cols != 32:
            raise ValueError(f"Se esperaban 32 columnas, pero se encontraron {num_cols}")

        col_names = ['idx_orig'] + [f'{coord}{i}' for i in range(1, 16) for coord in ['x', 'y']] + ['image_name']
        df.columns = col_names

        shapes = []
        image_names = []
        original_indices = []
        num_landmarks = 15

        for index, row in df.iterrows():
            coords = np.zeros((num_landmarks, 2), dtype=np.float32)
            valid_row = True
            try:
                for i in range(num_landmarks):
                    x = float(row[f'x{i+1}'])
                    y = float(row[f'y{i+1}'])
                    coords[i, 0] = x
                    coords[i, 1] = y
                # Validar si hay NaNs o Infs (importante para GPA)
                if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                     print(f"Advertencia: Coordenadas inválidas (NaN/Inf) en fila {index}. Saltando.")
                     continue
                shapes.append(coords)
                image_names.append(row['image_name'])
                original_indices.append(row['idx_orig'])
            except (KeyError, ValueError) as e:
                print(f"Advertencia: Error procesando fila {index} del CSV ({e}). Saltando.")
                continue

        print(f"Cargadas {len(shapes)} formas válidas desde {csv_path}")
        return shapes, image_names, original_indices

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de coordenadas: {csv_path}")
        return [], [], []
    except Exception as e:
        print(f"Error al leer o procesar el archivo CSV: {e}")
        return [], [], []

def save_aligned_coords(output_path, aligned_coords_list, image_names, original_indices):
    """Guarda las coordenadas alineadas en un nuevo archivo CSV."""
    data_to_save = []
    for i, coords_np in enumerate(aligned_coords_list):
        flat_coords = coords_np.flatten().tolist()
        if len(flat_coords) == 30:
             data_to_save.append([original_indices[i]] + flat_coords + [image_names[i]])
        else:
             print(f"Advertencia: Número inesperado de coordenadas ({len(flat_coords)}) para índice {i}. Saltando guardado de esta fila.")

    df_aligned = pd.DataFrame(data_to_save)
    df_aligned.to_csv(output_path, index=False, header=False)
    print(f"Coordenadas alineadas guardadas en: {output_path}")


def procrustes_gpa(shapes_list, max_iterations=1000, tolerance=1e-5):
    """
    Realiza Análisis Procrustes Generalizado (GPA) manual.
    Args:
        shapes_list: Lista de arrays NumPy (k, d), donde k=landmarks, d=dimensiones.
        max_iterations: Límite de iteraciones.
        tolerance: Umbral de convergencia para la diferencia entre formas medias.
    Returns:
        aligned_shapes: Lista de arrays NumPy (k, d) con las formas alineadas
                       (centradas en origen, escala normalizada a 1).
        mean_shape: Array NumPy (k, d) de la forma media final (centrada, escala 1).
    """
    n_shapes = len(shapes_list)
    if n_shapes == 0:
        return [], None
    n_landmarks, n_dims = shapes_list[0].shape

    # 1. Preprocesamiento: Centrar y normalizar escala (Tamaño Centroide = 1)
    processed_shapes = []
    for shape in shapes_list:
        centroid = np.mean(shape, axis=0)
        centered_shape = shape - centroid
        # Calcular Tamaño Centroide
        size = np.linalg.norm(centered_shape)
        if size < 1e-9: # Evitar división por cero
             # Si el tamaño es casi cero, la forma es un punto, mantenerlo centrado.
             # La escala no se puede normalizar significativamente.
             print("Advertencia: Forma con tamaño de centroide casi cero encontrada.")
             normalized_shape = centered_shape
        else:
            normalized_shape = centered_shape / size
        processed_shapes.append(normalized_shape)

    # 2. Inicialización: Usar la primera forma preprocesada como media inicial
    mean_shape = processed_shapes[0].copy()

    # 3. Iteración GPA
    for iteration in range(max_iterations):
        prev_mean_shape = mean_shape.copy()
        aligned_shapes_sum = np.zeros_like(mean_shape)
        current_aligned_shapes = [None] * n_shapes

        # Alinear cada forma a la media actual
        for i, shape in enumerate(processed_shapes):
            # Calcular rotación óptima (usando SVD) para alinear shape -> mean_shape
            cross_cov = shape.T @ mean_shape # Matriz kxd.T @ kxd -> dxd (2x15.T @ 15x2 -> 2x2)
            U, s, Vt = svd(cross_cov) # Vt es V transpuesta
            R = Vt.T @ U.T

            # Corrección de posible reflexión (si det(R) == -1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1 # Cambiar signo de la última fila de V^T
                R = Vt.T @ U.T

            # Aplicar rotación a la forma (preprocesada)
            aligned_shape = shape @ R
            current_aligned_shapes[i] = aligned_shape
            aligned_shapes_sum += aligned_shape

        # Calcular nueva media cruda
        new_mean_raw = aligned_shapes_sum / n_shapes

        # Normalizar la nueva media (centrar y escalar a tamaño 1)
        new_mean_centroid = np.mean(new_mean_raw, axis=0)
        new_mean_centered = new_mean_raw - new_mean_centroid
        new_mean_size = np.linalg.norm(new_mean_centered)

        if new_mean_size < 1e-9:
            print("Advertencia: La forma media colapsó a tamaño cero.")
            mean_shape = new_mean_centered # Mantener centrado
        else:
            mean_shape = new_mean_centered / new_mean_size

        # Comprobar convergencia (diferencia entre medias normalizadas)
        diff = np.linalg.norm(mean_shape - prev_mean_shape)
        #print(f"Iter {iteration + 1}, Diff: {diff:.6g}") # Descomentar para depurar
        if diff < tolerance:
            # print(f"GPA convergió en {iteration + 1} iteraciones.")
            break
    else: # Se ejecuta si el bucle termina sin 'break'
        print(f"Advertencia: GPA no convergió en {max_iterations} iteraciones (diff={diff:.6g}).")

    # Devolver las formas alineadas en la última iteración y la media final
    # IMPORTANTE: current_aligned_shapes están centradas y normalizadas a tamaño 1
    return current_aligned_shapes, mean_shape


# --- Carga de Datos ---
original_shapes_np, image_names_base, original_indices = load_shapes_from_csv(coords_file)

if not original_shapes_np:
    print("No se cargaron formas válidas. Terminando.")
    exit()

# --- Realizar Análisis Procrustes Generalizado ---
print("Realizando Análisis Procrustes Generalizado (GPA)...")
# aligned_shapes_final_norm: lista de arrays (15, 2), centrados, escala normalizada
# final_mean_shape: array (15, 2) de la forma media (centrada, escala normalizada)
aligned_shapes_final_norm, final_mean_shape = procrustes_gpa(original_shapes_np)
print("GPA completado.")

if not aligned_shapes_final_norm:
     print("GPA falló o no produjo resultados. Terminando.")
     exit()

# --- Aplicar Transformación a Imágenes ---
print("Aplicando transformaciones a las imágenes...")
processed_count = 0
skipped_count = 0

# Lista para guardar las coordenadas que realmente se usaron (por si alguna se omitió)
final_coords_to_save = []
final_names_to_save = []
final_indices_to_save = []


for i, image_name_base in enumerate(image_names_base):
    # Obtener puntos originales y alineados para esta imagen
    # Asegurarse de que el índice i todavía sea válido si se omitieron formas en la carga
    if i >= len(aligned_shapes_final_norm): break # Seguridad

    pts_orig = original_shapes_np[i].astype(np.float32)
    # Los puntos alineados son los normalizados devueltos por GPA
    pts_aligned = aligned_shapes_final_norm[i].astype(np.float32)

    # Cargar la imagen 64x64 correspondiente
    image_name = image_name_base + ".png"
    image_path = os.path.join(image_dir_64x64, image_name)

    if not os.path.exists(image_path):
        print(f"Advertencia: No se encontró imagen {image_path} para índice {i}. Saltando.")
        skipped_count += 1
        continue

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Advertencia: No se pudo cargar imagen {image_path} para índice {i}. Saltando.")
        skipped_count += 1
        continue

    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Estimar la transformación de similitud (escala, rot, tras)
    # que mapea los puntos originales a los puntos alineados por GPA.
    M = cv2.estimateAffinePartial2D(pts_orig, pts_aligned)[0]

    if M is None:
        print(f"Advertencia: No se pudo estimar la transformación para {image_name} (índice original {original_indices[i]}). Saltando.")
        skipped_count += 1
        continue

    # Aplicar la transformación M a la imagen completa
    img_aligned_procrustes = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)

    # Guardar la imagen transformada
    output_image_path = os.path.join(output_image_dir, image_name)
    cv2.imwrite(output_image_path, img_aligned_procrustes)

    # Añadir las coordenadas alineadas a la lista final
    final_coords_to_save.append(pts_aligned)
    final_names_to_save.append(image_name_base)
    final_indices_to_save.append(original_indices[i])

    processed_count += 1
    if processed_count % 50 == 0:
        print(f"Procesadas {processed_count} imágenes...")

# --- Guardar Coordenadas Alineadas ---
# Pasar solo los datos de las imágenes que SÍ se procesaron
save_aligned_coords(output_coords_file, final_coords_to_save, final_names_to_save, final_indices_to_save)

print("-" * 30)
print(f"Proceso completado.")
print(f"Imágenes procesadas y guardadas: {processed_count}")
print(f"Imágenes omitidas (errores/no encontradas/transformación fallida): {skipped_count}")
print(f"Imágenes alineadas por Procrustes guardadas en: {output_image_dir}")
print(f"Coordenadas alineadas por Procrustes guardadas en: {output_coords_file}")