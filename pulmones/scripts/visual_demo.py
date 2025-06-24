import os
import sys
import tkinter as tk
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

# --- Configuración de Rutas (AJUSTAR SI ES NECESARIO) ---
# Se asume que este script se encuentra en la misma carpeta que los otros
# scripts de alto nivel.
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")

if SRC_DIR_PULMONES not in sys.path:
    sys.path.append(SRC_DIR_PULMONES)

# --- Importaciones del Proyecto ---
from core.appearance_model import MultiLevelAppearanceModel
from core.asm_fitter import ASMFitter
from core.shape_model import ShapeModel


# --- Funciones de Ayuda para Warping (copiadas de process_and_warp_all.py) ---
def normalize_shape_to_box(shape, box_size):
    """Escala y traslada una forma para que encaje en una caja de tamaño `box_size`."""
    min_coords, max_coords = np.min(shape, axis=0), np.max(shape, axis=0)
    shape_dims = max_coords - min_coords
    shape_dims[shape_dims == 0] = 1
    scale = min((np.array(box_size) * 0.95) / shape_dims)
    scaled_shape = (shape - min_coords) * scale
    translation = (np.array(box_size) - np.max(scaled_shape, axis=0)) / 2.0
    return scaled_shape + translation


def warp_image(image, source_points, target_points, output_shape):
    """Realiza un warping de la imagen basado en triangulación."""
    tri_dest = Delaunay(target_points)
    h, w = output_shape
    is_color = image.ndim == 3
    warped_image = np.zeros((h, w, 3) if is_color else (h, w), dtype=np.uint8)

    for simplex in tri_dest.simplices:
        src_tri = source_points[simplex].astype(np.float32)
        dst_tri = target_points[simplex].astype(np.float32)

        # Matriz de transformación afín para este triángulo
        M = cv2.getAffineTransform(src_tri, dst_tri)

        # Encontrar la caja delimitadora del triángulo de destino
        x, y, w_box, h_box = cv2.boundingRect(dst_tri)

        # Recortar la región de interés (ROI)
        roi_dst = np.zeros_like(
            image, shape=(h_box, w_box, 3) if is_color else (h_box, w_box)
        )

        # Ajustar los puntos del triángulo al sistema de coordenadas de la ROI
        dst_tri_roi = dst_tri - np.array([x, y])

        # Crear una máscara para el triángulo de destino
        mask = np.zeros((h_box, w_box), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri_roi.astype(np.int32), 1, 16)

        # Aplicar la transformación afín
        warped_roi = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

        # Extraer la ROI de la imagen warpeada
        roi_warped = warped_roi[y : y + h_box, x : x + w_box]

        # Combinar usando la máscara
        cv2.copyTo(roi_warped, mask, warped_image[y : y + h_box, x : x + w_box])

    return warped_image


def run_demo_on_image(
    image_path, shape_model, appearance_model, fitter_params, connections, output_size
):
    """
    Ejecuta el pipeline completo de ASM en una sola imagen y visualiza los resultados.
    """
    print(f"Procesando imagen: {os.path.basename(image_path)}")

    # 1. Cargar la imagen
    image_orig_color = cv2.imread(image_path)
    if image_orig_color is None:
        print("Error: No se pudo cargar la imagen.")
        return
    image_gray = cv2.cvtColor(image_orig_color, cv2.COLOR_BGR2GRAY)

    # 2. Instanciar el Fitter y ejecutar el ajuste
    print("Inicializando ASM Fitter y ajustando el modelo...")
    asm_fitter = ASMFitter(shape_model, appearance_model, fitting_params=fitter_params)
    predicted_shape, _ = asm_fitter.fit_model_to_image(image_gray, verbose=True)

    if predicted_shape is None:
        print("El ajuste del modelo falló.")
        return

    print("Ajuste completado. Generando visualizaciones...")

    # 3. Preparar para el Warping
    mean_shape = shape_model.get_mean_shape_procrustes()
    target_points = normalize_shape_to_box(mean_shape, output_size)
    warped_image = warp_image(
        image_orig_color,
        predicted_shape,
        target_points,
        (output_size[1], output_size[0]),
    )

    # 4. Crear la visualización de múltiples paneles
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"Demostración Visual del Proceso ASM para: {os.path.basename(image_path)}",
        fontsize=20,
    )
    plt.style.use("default")

    # --- Panel 1: Imagen Original ---
    axes[0, 0].imshow(cv2.cvtColor(image_orig_color, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Imagen de Entrada Original", fontsize=14)
    axes[0, 0].axis("off")

    # --- Panel 2: Resultado del Ajuste ASM ---
    axes[0, 1].imshow(image_gray, cmap="gray")
    for p1, p2 in connections:
        axes[0, 1].plot(
            [predicted_shape[p1, 0], predicted_shape[p2, 0]],
            [predicted_shape[p1, 1], predicted_shape[p2, 1]],
            "r-",
            lw=1.5,
        )
    axes[0, 1].plot(
        predicted_shape[:, 0],
        predicted_shape[:, 1],
        "r.",
        markersize=6,
        label="Predicción ASM",
    )
    axes[0, 1].set_title("2. Landmarks Predichos por ASM", fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].axis("off")

    # --- Panel 3: Triangulación de Delaunay ---
    axes[0, 2].imshow(image_gray, cmap="gray")
    axes[0, 2].triplot(
        predicted_shape[:, 0],
        predicted_shape[:, 1],
        Delaunay(predicted_shape).simplices,
        "c-",
        lw=1.0,
        label="Malla de Warping",
    )
    axes[0, 2].plot(predicted_shape[:, 0], predicted_shape[:, 1], "c.", markersize=4)
    axes[0, 2].set_title("3. Malla de Origen para Warping", fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].axis("off")

    # --- Panel 4: Malla de Destino Canónica ---
    axes[1, 0].set_facecolor("black")
    axes[1, 0].triplot(
        target_points[:, 0],
        target_points[:, 1],
        Delaunay(target_points).simplices,
        "y-",
        lw=1.0,
        label="Malla Canónica",
    )
    axes[1, 0].plot(target_points[:, 0], target_points[:, 1], "yo", markersize=4)
    axes[1, 0].set_title("4. Malla de Destino (Forma Media)", fontsize=14)
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend()
    axes[1, 0].axis("off")

    # --- Panel 5: Resultado Normalizado (Warped) ---
    axes[1, 1].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Imagen Normalizada (Warped)", fontsize=14)
    axes[1, 1].axis("off")

    # --- Panel 6: Texto Explicativo ---
    axes[1, 2].axis("off")
    info_text = (
        "Proceso de Demostración:\n\n"
        "1. Imagen Original: La entrada al sistema.\n\n"
        "2. Predicción ASM: El modelo encuentra los\n"
        "   landmarks pulmonares clave usando una\n"
        "   búsqueda iterativa coarse-to-fine.\n\n"
        "3. Malla de Origen: Se crea una malla\n"
        "   (Triangulación de Delaunay) sobre los\n"
        "   puntos predichos.\n\n"
        "4. Malla de Destino: La forma media del\n"
        "   dataset, que sirve como referencia\n"
        "   canónica.\n\n"
        "5. Imagen Normalizada: La imagen original\n"
        "   es deformada (warped) para que la malla\n"
        "   de origen coincida con la de destino,\n"
        "   logrando la normalización de la forma."
    )
    axes[1, 2].text(
        0.5, 0.5, info_text, ha="center", va="center", fontsize=12, wrap=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    """
    Función principal que carga los modelos y lanza el selector de archivos.
    """
    # --- Configuración del Demo ---
    models_dir = os.path.join(PROJECT_ROOT_DIR, "pulmones", "models")
    shape_model_path = os.path.join(models_dir, "lung_shape_model_full_aug.pkl")
    appearance_model_base_path = os.path.join(
        models_dir, "lung_appearance_model_full_aug"
    )
    output_size = (256, 256)

    # Conexiones para una visualización completa y correcta
    contour_connections = [
        (0, 12),
        (12, 3),
        (3, 5),
        (5, 7),
        (7, 14),
        (14, 1),
        (1, 13),
        (13, 6),
        (6, 4),
        (4, 2),
        (2, 11),
        (11, 0),
    ]
    midline_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
    all_connections = contour_connections + midline_connections

    # Parámetros de ajuste probados que dieron los mejores resultados
    best_fitting_params = {
        "iterations_per_level": [50, 28, 8],
        "profile_search_length_px": 10,
        "contour_indices_ordered": [0, 12, 3, 5, 7, 14, 1, 13, 6, 4, 2, 11],
    }

    # 1. Cargar los modelos una sola vez
    print("Cargando modelos de Forma y Apariencia...")
    shape_model = ShapeModel.load(shape_model_path)
    appearance_model = MultiLevelAppearanceModel.load(appearance_model_base_path)

    if not shape_model or not appearance_model:
        print(
            "Error: No se pudieron cargar los modelos. Asegúrate de que los archivos existen en la ruta correcta."
        )
        return

    print("Modelos cargados exitosamente.")

    # 2. Configurar la GUI para seleccionar archivo
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter

    while True:
        # Abrir el diálogo para seleccionar un archivo de imagen
        image_path = filedialog.askopenfilename(
            title="Selecciona una radiografía para procesar",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )

        if not image_path:
            print("No se seleccionó ninguna imagen. Saliendo del programa.")
            break

        # Ejecutar el demo con la imagen seleccionada
        run_demo_on_image(
            image_path,
            shape_model,
            appearance_model,
            best_fitting_params,
            all_connections,
            output_size,
        )

        # Preguntar si desea procesar otra imagen
        if not tk.messagebox.askyesno("Continuar", "¿Deseas procesar otra imagen?"):
            print("Proceso finalizado por el usuario.")
            break


if __name__ == "__main__":
    main()
