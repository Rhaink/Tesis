import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import Delaunay
from tqdm import tqdm

# --- Configuración de Rutas ---
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")

if SRC_DIR_PULMONES not in sys.path:
    sys.path.append(SRC_DIR_PULMONES)

# --- Importaciones de nuestro proyecto ---
from core.appearance_model import MultiLevelAppearanceModel
from core.asm_fitter import ASMFitter
from core.shape_model import ShapeModel
from utils import asm_utils


# --- Funciones Auxiliares (Verificadas y sin cambios) ---
def calculate_point_to_point_error(predicted_shape, ground_truth_shape):
    return np.mean(np.linalg.norm(predicted_shape - ground_truth_shape, axis=1))


def normalize_shape_to_box(shape, box_size):
    min_coords, max_coords = np.min(shape, axis=0), np.max(shape, axis=0)
    shape_dims = max_coords - min_coords
    shape_dims[shape_dims == 0] = 1
    scale = min((np.array(box_size) * 0.95) / shape_dims)
    scaled_shape = (shape - min_coords) * scale
    translation = (np.array(box_size) - np.max(scaled_shape, axis=0)) / 2.0
    return scaled_shape + translation


def warp_image(image, source_points, target_points, output_shape):
    tri_dest = Delaunay(target_points)
    h, w = output_shape
    is_color = image.ndim == 3
    warped_image = np.zeros((h, w, 3) if is_color else (h, w), dtype=np.uint8)
    x_out, y_out = np.meshgrid(np.arange(w), np.arange(h))
    output_pixel_coords = np.vstack((x_out.ravel(), y_out.ravel())).T
    simplex_indices = tri_dest.find_simplex(output_pixel_coords)
    transforms = [
        cv2.getAffineTransform(
            target_points[s].astype(np.float32), source_points[s].astype(np.float32)
        )
        for s in tri_dest.simplices
    ]
    h_orig, w_orig = image.shape[:2]
    x_orig, y_orig = np.arange(w_orig), np.arange(h_orig)
    if is_color:
        interpolators = [
            RectBivariateSpline(y_orig, x_orig, image[:, :, c], kx=1, ky=1)
            for c in range(3)
        ]
    else:
        interpolator = RectBivariateSpline(y_orig, x_orig, image, kx=1, ky=1)
    for i, simplex in enumerate(tri_dest.simplices):
        pixels_in_simplex = output_pixel_coords[simplex_indices == i]
        if len(pixels_in_simplex) == 0:
            continue
        M = transforms[i]
        pixels_homog = np.hstack(
            (pixels_in_simplex, np.ones((len(pixels_in_simplex), 1)))
        )
        source_coords = (M @ pixels_homog.T).T
        if is_color:
            pixel_colors = np.zeros((len(source_coords), 3))
            for c in range(3):
                pixel_colors[:, c] = interpolators[c].ev(
                    source_coords[:, 1], source_coords[:, 0]
                )
        else:
            pixel_colors = interpolator.ev(source_coords[:, 1], source_coords[:, 0])
        warped_image[pixels_in_simplex[:, 1], pixels_in_simplex[:, 0]] = np.clip(
            pixel_colors, 0, 255
        ).astype(np.uint8)
    return warped_image


# ==============================================================================
# === SCRIPT PRINCIPAL DE PROCESAMIENTO INTEGRAL ===
# ==============================================================================
def main():
    print(
        "--- Proceso Integral: Evaluación, Clasificación y Warping (v4 Final y Estable) ---"
    )

    # --- Configuración ---
    models_dir = os.path.join(PROJECT_ROOT_DIR, "pulmones", "models")
    shape_model_path = os.path.join(models_dir, "lung_shape_model_full_aug.pkl")
    appearance_model_base_path = os.path.join(
        models_dir, "lung_appearance_model_full_aug"
    )
    test_coords_csv = os.path.join(
        PROJECT_ROOT_DIR, "coordenadas", "coordenadas_prueba_strict_curated_A15.csv"
    )
    images_base_dir = os.path.join(PROJECT_ROOT_DIR, "COVID-19_Radiography_Dataset")
    num_landmarks = 15
    output_base_dir = os.path.join(
        PROJECT_ROOT_DIR, "pulmones", "results", "final_classified_output_stable"
    )
    OUTPUT_SIZE = (512, 512)

    # --- Definiciones Clave para Reproducibilidad ---
    contour_indices_ordered = [0, 12, 3, 5, 7, 14, 1, 13, 6, 4, 2, 11]
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

    # ### INICIO DE LA CORRECCIÓN CRÍTICA ###
    # Esta es la configuración EXACTA que produjo los mejores resultados.
    # NO debe ser modificada.
    best_fitting_params = {
        "iterations_per_level": [50, 28, 8],
        "profile_search_length_px": 10,
        "contour_indices_ordered": contour_indices_ordered,  # USAR LA LISTA COMPLETA
    }
    # ### FIN DE LA CORRECCIÓN CRÍTICA ###

    # Crear estructura de directorios
    categories = {
        "error_menor_a_10px": (0, 10),
        "error_entre_10_y_20px": (10, 20),
        "error_mayor_a_20px": (20, float("inf")),
    }
    for cat_name in categories.keys():
        os.makedirs(
            os.path.join(output_base_dir, cat_name, "1_evaluation_plots"), exist_ok=True
        )
        os.makedirs(
            os.path.join(output_base_dir, cat_name, "2_warping_visualization"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(output_base_dir, cat_name, "3_warped_normalized_only"),
            exist_ok=True,
        )

    # Cargar modelos y configurar Fitter
    print("Cargando modelos y configurando Fitter con parámetros estables...")
    shape_model = ShapeModel.load(shape_model_path)
    appearance_model = MultiLevelAppearanceModel.load(appearance_model_base_path)
    if not shape_model or not appearance_model:
        return
    asm_fitter = ASMFitter(
        shape_model, appearance_model, fitting_params=best_fitting_params
    )

    # Cargar datos y preparar forma de destino
    gt_shapes, image_names = asm_utils.load_landmarks(
        test_coords_csv, num_landmarks=num_landmarks
    )
    mean_shape = shape_model.get_mean_shape_procrustes()
    target_points = normalize_shape_to_box(mean_shape, OUTPUT_SIZE)

    # Guardar malla de destino
    plt.figure(figsize=(6, 6))
    plt.triplot(
        target_points[:, 0],
        target_points[:, 1],
        Delaunay(target_points).simplices,
        "k-",
        lw=0.5,
    )
    plt.plot(target_points[:, 0], target_points[:, 1], "bo", markersize=3)
    plt.title("Malla de Destino Canónica")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.savefig(os.path.join(output_base_dir, "0_target_mean_shape_mesh.png"))
    plt.close()

    # Bucle principal de procesamiento
    for i, img_name in enumerate(tqdm(image_names, desc="Procesando imágenes")):
        gt_shape_current = gt_shapes[i]
        img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
        if not img_path:
            continue
        image_orig = cv2.imread(img_path)
        if image_orig is None:
            continue
        image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
        predicted_shape, _ = asm_fitter.fit_model_to_image(image_gray)
        if predicted_shape is None:
            continue

        h, w = image_gray.shape
        gt_shape_scaled = gt_shape_current.copy() * [w / 64.0, h / 64.0]
        error = calculate_point_to_point_error(predicted_shape, gt_shape_scaled)
        output_category_dir = None
        for cat, (min_e, max_e) in categories.items():
            if min_e <= error < max_e:
                output_category_dir = os.path.join(output_base_dir, cat)
                break
        if not output_category_dir:
            continue
        base_name = os.path.basename(img_name)
        output_filename = f"{base_name}.png"

        # Guardar Plot de Evaluación con contornos correctos
        plt.figure(figsize=(8, 8))
        plt.imshow(image_gray, cmap="gray")
        for p1, p2 in all_connections:
            plt.plot(
                [gt_shape_scaled[p1, 0], gt_shape_scaled[p2, 0]],
                [gt_shape_scaled[p1, 1], gt_shape_scaled[p2, 1]],
                "g-",
                lw=1,
            )
        plt.plot(gt_shape_scaled[:, 0], gt_shape_scaled[:, 1], "g.", markersize=5)
        for p1, p2 in all_connections:
            plt.plot(
                [predicted_shape[p1, 0], predicted_shape[p2, 0]],
                [predicted_shape[p1, 1], predicted_shape[p2, 1]],
                "r-",
                lw=1,
            )
        plt.plot(predicted_shape[:, 0], predicted_shape[:, 1], "r.", markersize=5)
        plt.title(f"Resultado para {base_name}\nError: {error:.2f} px")
        handles = [
            plt.Line2D([], [], color="g", ls="-", marker=".", label="Ground Truth"),
            plt.Line2D([], [], color="r", ls="-", marker=".", label="ASM Predicción"),
        ]
        plt.legend(handles=handles)
        plt.axis("off")
        plt.savefig(
            os.path.join(output_category_dir, "1_evaluation_plots", output_filename)
        )
        plt.close()

        # Realizar Warping y guardar
        warped_image = warp_image(
            image_orig, predicted_shape, target_points, (OUTPUT_SIZE[1], OUTPUT_SIZE[0])
        )
        cv2.imwrite(
            os.path.join(
                output_category_dir, "3_warped_normalized_only", output_filename
            ),
            warped_image,
        )

        # Guardar Visualización del Proceso de Warping
        fig_warp, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB))
        axes[0].triplot(
            predicted_shape[:, 0],
            predicted_shape[:, 1],
            Delaunay(predicted_shape).simplices,
            "r-",
            lw=0.8,
        )
        axes[0].plot(predicted_shape[:, 0], predicted_shape[:, 1], "go", markersize=3)
        axes[0].set_title("1. Original con Malla Predicha")
        axes[0].axis("off")
        axes[1].triplot(
            target_points[:, 0],
            target_points[:, 1],
            Delaunay(target_points).simplices,
            "b-",
            lw=0.8,
        )
        axes[1].plot(target_points[:, 0], target_points[:, 1], "yo", markersize=3)
        axes[1].set_title("2. Malla de Destino (Canónica)")
        axes[1].axis("off")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].invert_yaxis()
        axes[2].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title("3. Resultado Normalizado")
        axes[2].axis("off")
        fig_warp.suptitle(f"Proceso de Warping para {base_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            os.path.join(
                output_category_dir, "2_warping_visualization", output_filename
            )
        )
        plt.close(fig_warp)

    print("\n--- Proceso Integral Finalizado ---")


if __name__ == "__main__":
    main()
