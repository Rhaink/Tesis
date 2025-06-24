import os
import sys

import cv2
import numpy as np

# --- Configuración de Rutas ---
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")

if SRC_DIR_PULMONES not in sys.path:
    sys.path.append(SRC_DIR_PULMONES)

from core.appearance_model import MultiLevelAppearanceModel
from core.shape_model import ShapeModel
from utils import asm_utils

# ==============================================================================
# === FUNCIONES DE AUMENTACIÓN ===
# ==============================================================================


def generate_synthetic_shapes(
    initial_shape_model, num_shapes_to_generate, std_dev_limit=2.0
):
    if not initial_shape_model._is_trained:
        raise ValueError("Modelo de forma no entrenado.")
    eigenvalues = initial_shape_model.get_eigenvalues()
    num_components = len(eigenvalues)
    synthetic_shapes = []
    for _ in range(num_shapes_to_generate):
        random_b = np.zeros(num_components)
        for k in range(num_components):
            std_dev_k = np.sqrt(eigenvalues[k])
            random_weight = np.clip(
                np.random.normal(0, 1), -std_dev_limit, std_dev_limit
            )
            random_b[k] = random_weight * std_dev_k
        synthetic_shapes.append(initial_shape_model.reconstruct_shape(random_b))
    return np.array(synthetic_shapes)


def augment_intensity(image):
    augmented_images = []
    # 1. Ajuste de Contraste y Brillo
    for alpha in [0.8, 1.2]:
        for beta in [-20, 20]:
            aug_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            augmented_images.append(aug_img)
    # 2. Ruido Gaussiano
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    augmented_images.append(cv2.add(image, noise))
    # 3. Desenfoque Gaussiano
    augmented_images.append(cv2.GaussianBlur(image, (5, 5), 0))
    return augmented_images


# ==============================================================================
# === SCRIPT PRINCIPAL ===
# ==============================================================================


def main():
    print("--- Entrenamiento ASM con Aumentación COMPLETA (Forma + Apariencia) ---")

    # --- Parámetros ---
    coords_csv = os.path.join(
        PROJECT_ROOT_DIR, "coordenadas", "coordenadas_entrenamiento_1.csv"
    )
    images_base_dir = os.path.join(PROJECT_ROOT_DIR, "COVID-19_Radiography_Dataset")
    output_dir = os.path.join(PROJECT_ROOT_DIR, "pulmones", "models")

    shape_model_filename = "lung_shape_model_full_aug.pkl"
    appearance_model_base_filename = "lung_appearance_model_full_aug"

    NUM_SYNTHETIC_SHAPES = 2000
    num_landmarks = 15
    pca_n_components = 0.99
    pyramid_levels = 3
    profile_length_px = 21
    profile_num_points = 22
    contour_indices_ordered = [0, 12, 3, 5, 7, 14, 1, 13, 6, 4, 2, 11]

    os.makedirs(output_dir, exist_ok=True)

    # 1. Cargar datos originales
    print(f"\nCargando landmarks originales desde: {coords_csv}...")
    original_shapes, original_image_names = asm_utils.load_landmarks(
        coords_csv, num_landmarks=num_landmarks
    )

    # --- MODELO DE FORMA CON AUMENTACIÓN ---
    print("\n--- Entrenando Modelo de Forma con Aumentación ---")

    # Paso 1: Entrenar un modelo base para generar formas
    print("Paso 1/3: Entrenando modelo de forma inicial...")
    initial_shape_model = ShapeModel(num_landmarks=num_landmarks)
    initial_shape_model.train(original_shapes, pca_n_components=0.98)
    if not initial_shape_model._is_trained:
        print("Fallo al entrenar el modelo de forma inicial. Abortando.")
        return

    # Paso 2: Generar formas sintéticas
    print("Paso 2/3: Generando formas sintéticas...")
    synthetic_shapes = generate_synthetic_shapes(
        initial_shape_model, NUM_SYNTHETIC_SHAPES
    )

    # Paso 3: Combinar y entrenar el modelo final
    print("Paso 3/3: Entrenando modelo de forma final con dataset aumentado...")
    # Las formas sintéticas ya están en el espacio Procrustes. No necesitan alineación.
    # Solo necesitamos alinear las originales una vez.
    aligned_original_shapes, _ = asm_utils.generalized_procrustes_analysis(
        original_shapes
    )
    augmented_shapes_dataset = np.concatenate(
        [aligned_original_shapes, synthetic_shapes], axis=0
    )

    final_shape_model = ShapeModel(num_landmarks=num_landmarks)
    final_shape_model.train(augmented_shapes_dataset, pca_n_components=pca_n_components)
    if not final_shape_model._is_trained:
        print("Fallo al entrenar el modelo de forma final. Abortando.")
        return

    shape_model_path = os.path.join(output_dir, shape_model_filename)
    final_shape_model.save(shape_model_path)
    print(f"Modelo de Forma aumentado guardado en: {shape_model_path}")

    # --- MODELO DE APARIENCIA CON AUMENTACIÓN ---
    print("\n--- Entrenando Modelo de Apariencia con Aumentación de Intensidad ---")

    augmented_images_list = []
    corresponding_shapes_list = []

    print("Generando dataset de apariencia aumentado...")
    for i, img_name in enumerate(original_image_names):
        img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
        if not img_path:
            continue
        original_image = asm_utils.load_image_grayscale(img_path)
        if original_image is None:
            continue
        original_shape = original_shapes[i]

        augmented_images_list.append(original_image)
        corresponding_shapes_list.append(original_shape)

        intensity_augmented_imgs = augment_intensity(original_image)
        for aug_img in intensity_augmented_imgs:
            augmented_images_list.append(aug_img)
            corresponding_shapes_list.append(original_shape)

    print(f"Dataset de apariencia original: {len(original_image_names)} imágenes.")
    print(f"Dataset de apariencia aumentado: {len(augmented_images_list)} imágenes.")

    shapes_for_appearance_training = np.array(corresponding_shapes_list)

    appearance_model_instance = MultiLevelAppearanceModel()
    profile_params_dict = {
        "length": profile_length_px,
        "num_points": profile_num_points,
    }

    appearance_model_instance.train(
        images_list=augmented_images_list,
        shapes_list_orig_coords=shapes_for_appearance_training,
        num_levels=pyramid_levels,
        num_landmarks=num_landmarks,
        profile_params=profile_params_dict,
        contour_indices_ordered=contour_indices_ordered,
    )

    if not appearance_model_instance._is_trained:
        print("Fallo en el entrenamiento del Modelo de Apariencia. Abortando.")
        return

    appearance_model_base_path = os.path.join(
        output_dir, appearance_model_base_filename
    )
    appearance_model_instance.save(appearance_model_base_path)

    print("\n--- Entrenamiento del Modelo ASM con Aumentación COMPLETA Finalizado ---")


if __name__ == "__main__":
    main()
