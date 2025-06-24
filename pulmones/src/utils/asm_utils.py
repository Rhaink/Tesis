import os

import cv2
import numpy as np
import pandas as pd
from numba import njit
from scipy.interpolate import RectBivariateSpline
from sklearn.decomposition import PCA

USE_NUMBA_PROFILE = True


def load_landmarks(csv_path, num_landmarks=15):
    df = pd.read_csv(csv_path)
    coord_cols = df.columns[1 : (2 * num_landmarks + 1)]
    if df[coord_cols].isnull().values.any():
        df.dropna(subset=coord_cols, inplace=True)
    coords_data = df.iloc[:, 1 : (2 * num_landmarks + 1)].values
    image_names = df.iloc[:, (2 * num_landmarks + 1)].tolist()
    num_shapes = coords_data.shape[0]
    shapes = coords_data.reshape((num_shapes, num_landmarks, 2))
    if np.any(np.isnan(shapes)):
        valid_indices = [i for i, s in enumerate(shapes) if not np.any(np.isnan(s))]
        shapes = shapes[valid_indices]
        image_names = [image_names[i] for i in valid_indices]
    return shapes, image_names


def procrustes_normalize_shape(shape):
    if np.any(np.isnan(shape)):
        return shape
    centroid = np.mean(shape, axis=0)
    centered_shape = shape - centroid
    norm = np.linalg.norm(centered_shape, "fro")
    if norm < 1e-9:
        return centered_shape
    return centered_shape / norm


def procrustes_align_shape(shape_to_align, reference_shape):
    if np.any(np.isnan(shape_to_align)) or np.any(np.isnan(reference_shape)):
        return shape_to_align
    M = shape_to_align.T @ reference_shape
    try:
        U, S, Vt = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return shape_to_align
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt_corrected = Vt.copy()
        Vt_corrected[-1, :] *= -1
        R = Vt_corrected.T @ U.T
    return shape_to_align @ R


def generalized_procrustes_analysis(shapes, max_iters=100, tolerance=1e-6):
    num_initial_shapes = shapes.shape[0]
    valid_indices = [i for i, s in enumerate(shapes) if not np.any(np.isnan(s))]
    if len(valid_indices) < num_initial_shapes:
        if len(valid_indices) < 2:
            return np.array([]), np.array([])
        shapes = shapes[valid_indices]
    num_shapes = shapes.shape[0]
    if num_shapes == 0:
        return np.array([]), np.array([])
    normalized_shapes = np.array([procrustes_normalize_shape(s) for s in shapes])
    if np.any(np.isnan(normalized_shapes)):
        return np.array([]), np.array([])
    mean_shape_current = np.mean(normalized_shapes, axis=0)
    mean_shape_current = procrustes_normalize_shape(mean_shape_current)
    if np.any(np.isnan(mean_shape_current)):
        return np.array([]), np.array([])
    aligned_shapes_iter = normalized_shapes.copy()
    for iteration in range(max_iters):
        for i in range(num_shapes):
            aligned_shapes_iter[i] = procrustes_align_shape(
                normalized_shapes[i], mean_shape_current
            )
        if np.any(np.isnan(aligned_shapes_iter)):
            return aligned_shapes_iter, mean_shape_current
        mean_shape_new = np.mean(aligned_shapes_iter, axis=0)
        mean_shape_new = procrustes_normalize_shape(mean_shape_new)
        if np.any(np.isnan(mean_shape_new)):
            return aligned_shapes_iter, mean_shape_current
        diff = np.linalg.norm(mean_shape_new - mean_shape_current, "fro")
        if diff < tolerance:
            mean_shape_current = mean_shape_new
            break
        mean_shape_current = mean_shape_new
    else:
        print(f"GPA no convergió después de {max_iters} iteraciones.")
    final_aligned_shapes = aligned_shapes_iter
    if np.any(np.isnan(final_aligned_shapes)):
        print("ADVERTENCIA en GPA: Las formas alineadas finales contienen NaNs.")
    if np.any(np.isnan(mean_shape_current)):
        print("ADVERTENCIA en GPA: La forma media final es NaN.")
    return final_aligned_shapes, mean_shape_current


def build_pca_model(aligned_shapes, n_components=0.98):
    if aligned_shapes.size == 0:
        return PCA(), np.array([])
    if np.any(np.isnan(aligned_shapes)):
        return PCA(), np.array([])
    num_shapes, num_landmarks, dims = aligned_shapes.shape
    if num_shapes < 2:
        return PCA(), np.array([])
    shape_vectors = aligned_shapes.reshape(num_shapes, -1)
    if isinstance(n_components, int):
        max_possible_components = min(num_shapes, shape_vectors.shape[1])
        if n_components > max_possible_components:
            n_components = max_possible_components
        if n_components <= 0:
            return PCA(), np.array([])
    pca = PCA(n_components=n_components)
    try:
        pca.fit(shape_vectors)
    except ValueError:
        return PCA(), np.array([])
    return pca, pca.mean_


def get_image_path(image_id_str, indices_df, base_dataset_path):
    normalized_id_str_for_parse = image_id_str.replace("\\", "/").lower()
    if normalized_id_str_for_parse.startswith("images/"):
        base_name_for_file = image_id_str.replace("\\", "/")[len("images/") :]
        category_key_for_map = normalized_id_str_for_parse[len("images/") :].split("-")[
            0
        ]
    else:
        base_name_for_file = image_id_str.replace("\\", "/")
        category_key_for_map = normalized_id_str_for_parse.split("-")[0]
    parts = base_name_for_file.split("-")
    if len(parts) < 2:
        return None
    folder_map = {
        "covid": "COVID",
        "normal": "Normal",
        "viral pneumonia": "Viral Pneumonia",
        "lung_opacity": "Lung_Opacity",
    }
    if category_key_for_map not in folder_map:
        return None
    actual_folder_name = folder_map[category_key_for_map]
    if not base_name_for_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        final_filename = f"{base_name_for_file}.png"
    else:
        final_filename = base_name_for_file
    path = os.path.join(base_dataset_path, actual_folder_name, "images", final_filename)
    return path if os.path.exists(path) else None


def load_image_grayscale(image_path):
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


@njit
def calculate_normals(shape_points, contour_indices_ordered, is_closed_contour=True):
    num_total_landmarks = shape_points.shape[0]
    normals = np.zeros_like(shape_points, dtype=np.float64)
    num_contour_points = len(contour_indices_ordered)
    if num_contour_points < 2:
        return normals
    for i in range(num_contour_points):
        current_lm_idx = contour_indices_ordered[i]
        if is_closed_contour:
            prev_lm_idx = contour_indices_ordered[
                (i - 1 + num_contour_points) % num_contour_points
            ]
            next_lm_idx = contour_indices_ordered[(i + 1) % num_contour_points]
        else:
            if i == 0:
                prev_lm_idx, next_lm_idx = (
                    current_lm_idx,
                    contour_indices_ordered[i + 1],
                )
            elif i == num_contour_points - 1:
                prev_lm_idx, next_lm_idx = (
                    contour_indices_ordered[i - 1],
                    current_lm_idx,
                )
            else:
                prev_lm_idx, next_lm_idx = (
                    contour_indices_ordered[i - 1],
                    contour_indices_ordered[i + 1],
                )
        p_prev, p_next = shape_points[prev_lm_idx], shape_points[next_lm_idx]
        if not is_closed_contour:
            if i == 0:
                tangent = shape_points[next_lm_idx] - shape_points[current_lm_idx]
            elif i == num_contour_points - 1:
                tangent = shape_points[current_lm_idx] - shape_points[prev_lm_idx]
            else:
                tangent = p_next - p_prev
        else:
            tangent = p_next - p_prev
        normal_vec = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        norm_mag = np.sqrt(normal_vec[0] ** 2 + normal_vec[1] ** 2)
        if norm_mag > 1e-9:
            normals[current_lm_idx, 0] = normal_vec[0] / norm_mag
            normals[current_lm_idx, 1] = normal_vec[1] / norm_mag
    return normals


@njit
def get_pixel_value_bilinear(image, x, y):
    h, w = image.shape
    if not (0 <= x < w - 1 and 0 <= y < h - 1):
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0.0
        return float(image[int(y), int(x)])
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1
    fx, fy = x - x1, y - y1
    q11, q12, q21, q22 = (
        float(image[y1, x1]),
        float(image[y2, x1]),
        float(image[y1, x2]),
        float(image[y2, x2]),
    )
    return (
        q11 * (1 - fx) * (1 - fy)
        + q21 * fx * (1 - fy)
        + q12 * (1 - fx) * fy
        + q22 * fx * fy
    )


### CAMBIO ###
# La función ahora devuelve (perfil_derivada_normalizado, fuerza_borde)
@njit
def _get_profile_numba(image, point, normal, length, num_points_on_profile):
    h, w = image.shape
    zeros_result = (np.zeros(num_points_on_profile, dtype=np.float64), 0.0)
    if h <= 1 or w <= 1:
        return zeros_result
    distances_from_center = np.linspace(
        -length / 2.0, length / 2.0, num_points_on_profile
    )
    raw_intensities_profile = np.zeros(num_points_on_profile, dtype=np.float64)
    for i in range(num_points_on_profile):
        dist = distances_from_center[i]
        sample_x, sample_y = point[0] + dist * normal[0], point[1] + dist * normal[1]
        raw_intensities_profile[i] = get_pixel_value_bilinear(image, sample_x, sample_y)
    if num_points_on_profile < 2:
        return zeros_result
    profile_derivative = np.zeros(num_points_on_profile, dtype=np.float64)
    if num_points_on_profile > 1:
        profile_derivative[0] = raw_intensities_profile[1] - raw_intensities_profile[0]
        profile_derivative[-1] = (
            raw_intensities_profile[-1] - raw_intensities_profile[-2]
        )
        for i in range(1, num_points_on_profile - 1):
            profile_derivative[i] = (
                raw_intensities_profile[i + 1] - raw_intensities_profile[i - 1]
            ) / 2.0

    ### CAMBIO ###
    # Calculamos la magnitud de la derivada (fuerza del borde)
    edge_strength = np.mean(np.abs(profile_derivative))

    mean_derivative, std_derivative = (
        np.mean(profile_derivative),
        np.std(profile_derivative),
    )
    if std_derivative < 1e-9:
        profile_derivative_norm = profile_derivative - mean_derivative
    else:
        profile_derivative_norm = (
            profile_derivative - mean_derivative
        ) / std_derivative
    return profile_derivative_norm, edge_strength


### CAMBIO ###
# La versión SciPy también devuelve (perfil_derivada_normalizado, fuerza_borde)
def _get_profile_scipy(image, point, normal, length, num_points_on_profile):
    h, w = image.shape
    zeros_result = (np.zeros(num_points_on_profile), 0.0)
    if h <= 1 or w <= 1:
        return zeros_result
    y_coords, x_coords = np.arange(h), np.arange(w)
    try:
        interpolator = RectBivariateSpline(y_coords, x_coords, image, kx=1, ky=1)
    except ValueError:
        return zeros_result
    distances = np.linspace(-length / 2.0, length / 2.0, num_points_on_profile)
    sample_x, sample_y = (
        point[0] + distances * normal[0],
        point[1] + distances * normal[1],
    )
    try:
        raw_profile = interpolator(sample_y, sample_x, grid=False)
    except Exception:
        return zeros_result
    if len(raw_profile) < 2:
        return zeros_result
    profile_derivative = np.gradient(raw_profile)

    ### CAMBIO ###
    edge_strength = np.mean(np.abs(profile_derivative))

    mean_derivative, std_derivative = (
        np.mean(profile_derivative),
        np.std(profile_derivative),
    )
    if std_derivative < 1e-9:
        profile_derivative_norm = profile_derivative - mean_derivative
    else:
        profile_derivative_norm = (
            profile_derivative - mean_derivative
        ) / std_derivative
    return profile_derivative_norm, edge_strength


def get_profile(image, point, normal, length, num_points_on_profile):
    """Wrapper para la extracción de perfiles que devuelve (perfil_norm, fuerza_borde)."""
    if USE_NUMBA_PROFILE:
        # Asegurarse que los tipos de datos son correctos para Numba
        image_f64 = image.astype(np.float64)
        point_f64 = point.astype(np.float64)
        normal_f64 = normal.astype(np.float64)
        return _get_profile_numba(
            image_f64, point_f64, normal_f64, float(length), int(num_points_on_profile)
        )
    else:
        return _get_profile_scipy(image, point, normal, length, num_points_on_profile)


@njit
def transform_points(points, R, t, s):
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return (s * (points @ R.T)) + t


@njit
def inverse_transform_points(points_img, R, t, s):
    if points_img.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if abs(s) < 1e-9:
        return (points_img - t) @ R
    return (1.0 / s) * ((points_img - t) @ R)


def create_image_pyramid(image, levels):
    pyramid = [image.copy()]
    current_image = image.copy()
    for _ in range(1, levels):
        if min(current_image.shape) < 4:
            break
        downscaled_image = cv2.pyrDown(current_image)
        pyramid.append(downscaled_image)
        current_image = downscaled_image
    return pyramid


def plot_shape(
    ax,
    shape_coords,
    connections=None,
    color="blue",
    marker=".",
    markersize=5,
    linestyle="-",
    linewidth=1,
    clear_ax=True,
    set_limits=None,
    title=None,
):
    if clear_ax:
        ax.clear()
    if shape_coords.size == 0:
        if title:
            ax.set_title(title)
        return
    xs, ys = shape_coords[:, 0], shape_coords[:, 1]
    ax.plot(
        xs,
        ys,
        marker=marker,
        linestyle="None",
        markersize=markersize,
        color=color,
        zorder=2,
    )
    if connections:
        for p1_idx, p2_idx in connections:
            if 0 <= p1_idx < len(xs) and 0 <= p2_idx < len(xs):
                ax.plot(
                    [xs[p1_idx], xs[p2_idx]],
                    [ys[p1_idx], ys[p2_idx]],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    color=color,
                    zorder=1,
                )
    if set_limits == "auto":
        ax.autoscale_view()
    elif set_limits == "equal_aspect_data":
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
    elif isinstance(set_limits, (list, tuple)) and len(set_limits) == 4:
        ax.set_xlim(set_limits[0], set_limits[1])
        ax.set_ylim(set_limits[2], set_limits[3])
    if title:
        ax.set_title(title)
