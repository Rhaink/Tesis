# align_images_with_scaling.py

import os
import pandas as pd
import numpy as np
import cv2
import math
import shutil

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
output_image_dir = os.path.join(base_dir, "dataset_aligned_maestro_1")
# Directorio de salida para el nuevo archivo de coordenadas
output_coords_file = os.path.join(base_dir, "coordenadas_aligned_maestro_1.csv")

IMG_SIZE = 64
IMG_CENTER = (IMG_SIZE / 2 - 0.5, IMG_SIZE / 2 - 0.5)

# --- NUEVO: Distancia objetivo para la escala ---
# Define qué tan 'larga' quieres que sea la línea P1-P2 en píxeles después de alinear.
# Puedes calcular el promedio en tu dataset o elegir un valor fijo.
# Ejemplo: Que la línea vertebral ocupe la mitad de la altura de la imagen.
TARGET_DISTANCE = 30.0

# --- Crear directorios de salida ---
os.makedirs(output_image_dir, exist_ok=True)

# --- Función para transformar puntos (sin cambios) ---
def transform_points(points, M):
    points_np = np.float32(points).reshape(-1, 1, 2)
    transformed_points_np = cv2.transform(points_np, M)
    return transformed_points_np.reshape(-1, 2).tolist()

# --- Cargar coordenadas (sin cambios) ---
try:
    df_coords = pd.read_csv(coords_file, header=None)
    num_cols = len(df_coords.columns)
    if num_cols == 32:
         col_names = ['idx_orig'] + [f'{coord}{i}' for i in range(1, 16) for coord in ['x', 'y']] + ['image_name']
         df_coords.columns = col_names
    else:
        print(f"Error: Se esperaban 32 columnas en {coords_file}, pero se encontraron {num_cols}.")
        exit()
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de coordenadas: {coords_file}")
    exit()
except Exception as e:
    print(f"Error al leer el archivo CSV: {e}")
    exit()

# --- Procesar cada imagen ---
aligned_data = []
print("Iniciando proceso de alineación con escalado...")

for index, row in df_coords.iterrows():
    image_name = row['image_name'] + ".png"
    image_path = os.path.join(image_dir_64x64, image_name)

    if not os.path.exists(image_path):
        print(f"Advertencia: No se encontró la imagen {image_path}. Saltando...")
        continue
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Advertencia: No se pudo cargar la imagen {image_path}. Saltando...")
        continue
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
         #print(f"Advertencia: La imagen {image_name} no es de {IMG_SIZE}x{IMG_SIZE}. Redimensionando...")
         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    points = []
    try:
        for i in range(1, 16):
            x = row[f'x{i}']
            y = row[f'y{i}']
            points.append([float(x), float(y)])
    except KeyError as e:
         print(f"Error: Falta la columna {e} en la fila {index} del CSV. Saltando imagen {image_name}")
         continue
    except ValueError as e:
         print(f"Error: Valor no numérico en coordenadas para la imagen {image_name} (fila {index}). Saltando. ({e})")
         continue

    # 1. Calcular transformación (Rotación + Escala)
    p1 = points[0]
    p2 = points[1]
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    midpoint = (mid_x, mid_y)

    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]

    # Calcular distancia actual
    current_distance = math.sqrt(delta_x**2 + delta_y**2)

    # Calcular factor de escala
    if current_distance == 0:
        scale_factor = 1.0 # Evitar división por cero, no escalar
    else:
        scale_factor = TARGET_DISTANCE / current_distance

    # Calcular ángulo actual
    if delta_x == 0 and delta_y == 0:
        angle_rad = 0
    else:
        angle_rad = math.atan2(delta_x, delta_y)
    angle_deg = -math.degrees(angle_rad)

    # 2. Aplicar Rotación Y Escala alrededor del punto medio M
    # Matriz combinada de rotación y escala
    M_rot_scale = cv2.getRotationMatrix2D(midpoint, angle_deg, scale_factor) # <--- Se añade scale_factor

    # Rotar y escalar imagen
    img_rotated_scaled = cv2.warpAffine(img, M_rot_scale, (IMG_SIZE, IMG_SIZE),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

    # Rotar y escalar puntos
    points_rotated_scaled = transform_points(points, M_rot_scale)

    # 3. Calcular Traslación para centrar el punto medio transformado
    midpoint_rotated_scaled = points_rotated_scaled[0:2]
    mid_x_rot_scale = (midpoint_rotated_scaled[0][0] + midpoint_rotated_scaled[1][0]) / 2
    mid_y_rot_scale = (midpoint_rotated_scaled[0][1] + midpoint_rotated_scaled[1][1]) / 2

    tx = IMG_CENTER[0] - mid_x_rot_scale
    ty = IMG_CENTER[1] - mid_y_rot_scale
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])

    # 4. Aplicar Traslación
    img_aligned = cv2.warpAffine(img_rotated_scaled, M_trans, (IMG_SIZE, IMG_SIZE),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    points_aligned = transform_points(points_rotated_scaled, M_trans)

    # 5. Guardar resultados
    output_image_path = os.path.join(output_image_dir, image_name)
    cv2.imwrite(output_image_path, img_aligned)

    new_row_data = [row['idx_orig']]
    for p in points_aligned:
        new_row_data.extend([p[0], p[1]])
    new_row_data.append(row['image_name'])
    aligned_data.append(new_row_data)

    if (index + 1) % 50 == 0:
         print(f"Procesadas {index + 1}/{len(df_coords)} imágenes...")


# --- Crear y guardar el nuevo DataFrame de coordenadas ---
df_aligned = pd.DataFrame(aligned_data, columns=df_coords.columns)
df_aligned.to_csv(output_coords_file, index=False, header=False)

print("-" * 30)
print(f"Proceso completado.")
print(f"Imágenes alineadas y escaladas guardadas en: {output_image_dir}")
print(f"Nuevas coordenadas (alineadas y escaladas) guardadas en: {output_coords_file}")