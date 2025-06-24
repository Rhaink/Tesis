# align_images.py

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
output_image_dir = os.path.join(base_dir, "dataset/dataset_aligned_maestro_1")
# Directorio de salida para el nuevo archivo de coordenadas
output_coords_file = os.path.join(base_dir, "coordenadas/coordenadas_aligned_maestro_1.csv")

# Tamaño de la imagen (asumiendo cuadrado)
IMG_SIZE = 64
IMG_CENTER = (IMG_SIZE / 2 - 0.5, IMG_SIZE / 2 - 0.5) # Centro para cálculos (ej: 31.5, 31.5)

# --- Crear directorios de salida ---
os.makedirs(output_image_dir, exist_ok=True)

# --- Función para transformar puntos ---
def transform_points(points, M):
    """Aplica una matriz de transformación afín M (2x3) a una lista de puntos."""
    points_np = np.float32(points).reshape(-1, 1, 2) # Formato para cv2.transform
    transformed_points_np = cv2.transform(points_np, M)
    return transformed_points_np.reshape(-1, 2).tolist() # Devolver como lista de [x, y]

# --- Cargar coordenadas ---
try:
    df_coords = pd.read_csv(coords_file, header=None)
    # Asignar nombres de columna para claridad (asumiendo 1 índice + 15 puntos * 2 coords + 1 nombre)
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
print("Iniciando proceso de alineación...")

for index, row in df_coords.iterrows():
    image_name = row['image_name'] + ".png" # Asegúrate que la extensión sea correcta si no está en el CSV
    image_path = os.path.join(image_dir_64x64, image_name)

    # Cargar imagen en escala de grises
    if not os.path.exists(image_path):
        print(f"Advertencia: No se encontró la imagen {image_path}. Saltando...")
        continue

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Advertencia: No se pudo cargar la imagen {image_path}. Saltando...")
        continue

    # Verificar tamaño (opcional pero recomendado)
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
         print(f"Advertencia: La imagen {image_name} no es de {IMG_SIZE}x{IMG_SIZE}. Redimensionando...")
         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


    # Extraer coordenadas de los landmarks
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


    # 1. Calcular transformación
    p1 = points[0]
    p2 = points[1]

    # Punto medio
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    midpoint = (mid_x, mid_y)

    # Calcular ángulo actual respecto a la vertical (eje Y positivo hacia abajo)
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]

    # Evitar división por cero o atan2(0,0) si los puntos coinciden
    if delta_x == 0 and delta_y == 0:
        angle_rad = 0 # No hay línea definida, no rotar
    else:
        angle_rad = math.atan2(delta_x, delta_y) # Ángulo respecto a la vertical

    angle_deg = -math.degrees(angle_rad) # Ángulo para rotar (negativo para alinear con Y)

    # 2. Aplicar Rotación alrededor del punto medio M
    # Matriz de rotación que gira alrededor de 'midpoint'
    M_rot = cv2.getRotationMatrix2D(midpoint, angle_deg, 1.0)

    # Rotar imagen
    # Usar BORDER_CONSTANT con valor 0 (negro) para el fondo
    img_rotated = cv2.warpAffine(img, M_rot, (IMG_SIZE, IMG_SIZE),
                                 flags=cv2.INTER_LINEAR, # Interpolación bilineal
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

    # Rotar puntos
    points_rotated = transform_points(points, M_rot)

    # 3. Calcular Traslación para centrar el punto medio rotado
    midpoint_rotated = points_rotated[0:2] # Puntos 1 y 2 rotados
    mid_x_rot = (midpoint_rotated[0][0] + midpoint_rotated[1][0]) / 2
    mid_y_rot = (midpoint_rotated[0][1] + midpoint_rotated[1][1]) / 2

    tx = IMG_CENTER[0] - mid_x_rot
    ty = IMG_CENTER[1] - mid_y_rot

    # Matriz de traslación
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])

    # 4. Aplicar Traslación
    img_aligned = cv2.warpAffine(img_rotated, M_trans, (IMG_SIZE, IMG_SIZE),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

    points_aligned = transform_points(points_rotated, M_trans)

    # 5. Guardar resultados
    output_image_path = os.path.join(output_image_dir, image_name)
    cv2.imwrite(output_image_path, img_aligned)

    # Preparar datos para el nuevo CSV
    new_row_data = [row['idx_orig']] # Mantener el índice original si se desea
    for p in points_aligned:
        # Redondear o truncar si se prefieren coordenadas enteras, aunque float es más preciso
        new_row_data.extend([p[0], p[1]])
    new_row_data.append(row['image_name']) # Mantener el nombre original sin extensión
    aligned_data.append(new_row_data)

    if (index + 1) % 50 == 0: # Imprimir progreso cada 50 imágenes
         print(f"Procesadas {index + 1}/{len(df_coords)} imágenes...")


# --- Crear y guardar el nuevo DataFrame de coordenadas ---
df_aligned = pd.DataFrame(aligned_data, columns=df_coords.columns) # Usar los mismos nombres de columna
df_aligned.to_csv(output_coords_file, index=False, header=False) # Guardar sin índice ni cabecera, como el original

print("-" * 30)
print(f"Proceso completado.")
print(f"Imágenes alineadas guardadas en: {output_image_dir}")
print(f"Nuevas coordenadas guardadas en: {output_coords_file}")