import numpy as np
from utils import asm_utils

from .appearance_model import MultiLevelAppearanceModel
from .shape_model import ShapeModel


class ASMFitter:
    def __init__(self, shape_model, multi_level_appearance_model, fitting_params=None):
        if not isinstance(shape_model, ShapeModel) or not shape_model._is_trained:
            raise ValueError("Se requiere un ShapeModel entrenado.")
        if (
            not isinstance(multi_level_appearance_model, MultiLevelAppearanceModel)
            or not multi_level_appearance_model._is_trained
        ):
            raise ValueError("Se requiere un MultiLevelAppearanceModel entrenado.")

        self.shape_model = shape_model
        self.ml_app_model = multi_level_appearance_model

        default_params = {
            "iterations_per_level": [50, 30, 20],
            "profile_search_length_px": 10,
            "profile_search_steps": 21,
            "shape_param_limit_std": 3.0,
            "learning_rate_pose": 0.5,
            "learning_rate_shape": 0.5,
            "convergence_threshold": 1e-4,
            "contour_indices_ordered": list(range(self.shape_model.num_landmarks)),
            ### CAMBIO ###
            # Nuevo parámetro para ponderar la fuerza del borde.
            # Un valor pequeño (epsilon) para evitar división por cero.
            "edge_strength_epsilon": 1e-6,
        }
        self.params = {**default_params, **(fitting_params if fitting_params else {})}

        iters = self.params["iterations_per_level"]
        num_app_levels = self.ml_app_model.num_levels
        if len(iters) != num_app_levels:
            if len(iters) < num_app_levels:
                self.params["iterations_per_level"].extend(
                    [iters[-1]] * (num_app_levels - len(iters))
                )
            else:
                self.params["iterations_per_level"] = iters[:num_app_levels]

        self.model_profile_params = self.ml_app_model.profile_params
        if (
            not self.model_profile_params
            or "length" not in self.model_profile_params
            or "num_points" not in self.model_profile_params
        ):
            raise ValueError(
                "profile_params del modelo de apariencia no están completos."
            )

    def _initialize_pose_and_shape(self, image_shape_level0):
        img_h, img_w = image_shape_level0
        current_b = np.zeros(self.shape_model.pca_model.n_components_)
        current_R = np.eye(2)
        current_t = np.array([img_w / 2.0, img_h / 2.0])
        mean_shape_proc = self.shape_model.get_mean_shape_procrustes()
        min_c, max_c = np.min(mean_shape_proc, axis=0), np.max(mean_shape_proc, axis=0)
        model_w, model_h = max_c[0] - min_c[0], max_c[1] - min_c[1]
        if model_w < 1e-6:
            model_w = 1.0
        if model_h < 1e-6:
            model_h = 1.0
        scale_w = (img_w * 0.65) / model_w
        scale_h = (img_h * 0.65) / model_h
        current_s = (scale_w + scale_h) / 2.0
        return current_R, current_t, current_s, current_b

    def _propagate_params_to_next_level(
        self, R_prev, t_prev, s_prev, b_prev, prev_img_shape, next_img_shape
    ):
        R_next, b_next = R_prev.copy(), b_prev.copy()
        scale_x = next_img_shape[1] / prev_img_shape[1]
        scale_y = next_img_shape[0] / prev_img_shape[0]
        avg_scale = (scale_x + scale_y) / 2.0
        s_next = s_prev * avg_scale
        t_next = t_prev * avg_scale
        return R_next, t_next, s_next, b_next

    ### CAMBIO ###
    # La función de búsqueda de landmarks ahora usa la nueva función de coste.
    def _search_one_landmark(
        self, image_level, point_img, normal_img, appearance_model_lm
    ):
        if not appearance_model_lm._is_trained:
            return point_img

        img_h, img_w = image_level.shape
        best_pos_lm_img = point_img
        min_combined_cost = float("inf")
        epsilon = self.params["edge_strength_epsilon"]

        search_steps = np.linspace(
            -self.params["profile_search_length_px"],
            self.params["profile_search_length_px"],
            self.params["profile_search_steps"],
        )

        for step in search_steps:
            candidate_pos = point_img + step * normal_img
            if not (0 <= candidate_pos[0] < img_w and 0 <= candidate_pos[1] < img_h):
                continue

            # La función get_profile ahora devuelve dos valores
            sampled_profile_derivative, edge_strength = asm_utils.get_profile(
                image_level,
                candidate_pos,
                normal_img,
                self.model_profile_params["length"],
                self.model_profile_params["num_points"],
            )

            if sampled_profile_derivative is None or np.all(
                sampled_profile_derivative == 0
            ):
                continue

            mah_dist = appearance_model_lm.calculate_match_cost(
                sampled_profile_derivative
            )

            # NUEVA FUNCIÓN DE COSTE: Pondera la distancia de Mahalanobis por la inversa de la fuerza del borde.
            # Un borde fuerte (edge_strength alto) reduce drásticamente el coste.
            combined_cost = mah_dist / (edge_strength + epsilon)

            if combined_cost < min_combined_cost:
                min_combined_cost = combined_cost
                best_pos_lm_img = candidate_pos

        return best_pos_lm_img

    def _update_pose_and_shape(
        self, new_positions_img, R_current, t_current, s_current, b_current
    ):
        X_model_space_base = self.shape_model.get_mean_shape_procrustes()
        X_centroid, Y_centroid = (
            np.mean(X_model_space_base, axis=0),
            np.mean(new_positions_img, axis=0),
        )
        X_centered, Y_centered = (
            X_model_space_base - X_centroid,
            new_positions_img - Y_centroid,
        )
        H = X_centered.T @ Y_centered
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            R_new, s_new, t_new = R_current, s_current, t_current
        else:
            R_new = Vt.T @ U.T
            if np.linalg.det(R_new) < 0:
                Vt_corrected = Vt.copy()
                Vt_corrected[-1, :] *= -1
                R_new = Vt_corrected.T @ U.T
            X_centered_rotated = X_centered @ R_new.T
            numerator = np.sum(np.diag(Y_centered.T @ X_centered_rotated))
            denominator = np.sum(np.diag(X_centered_rotated.T @ X_centered_rotated))
            s_new = s_current if denominator < 1e-9 else numerator / denominator
            t_new = Y_centroid - (s_new * (X_centroid @ R_new.T))

        target_shape_procrustes = asm_utils.inverse_transform_points(
            new_positions_img, R_new, t_new, s_new
        )
        b_new_flat = self.shape_model.project_to_shape_space(target_shape_procrustes)

        eigenvalues = self.shape_model.get_eigenvalues()
        limit_std = self.params["shape_param_limit_std"]
        for k in range(len(b_new_flat)):
            if k < len(eigenvalues) and eigenvalues[k] > 0:
                limit_val_k = limit_std * np.sqrt(eigenvalues[k])
                b_new_flat[k] = np.clip(b_new_flat[k], -limit_val_k, limit_val_k)

        s_updated = (
            s_current * (1.0 - self.params["learning_rate_pose"])
            + s_new * self.params["learning_rate_pose"]
        )
        t_updated = (
            t_current * (1.0 - self.params["learning_rate_pose"])
            + t_new * self.params["learning_rate_pose"]
        )
        R_updated = R_new
        b_updated = (
            b_current * (1.0 - self.params["learning_rate_shape"])
            + b_new_flat * self.params["learning_rate_shape"]
        )
        return R_updated, t_updated, s_updated, b_updated

    def fit_model_to_image(
        self,
        image_original,
        initial_pose_params=None,
        verbose=False,
        debug_plot_fn=None,
    ):
        if image_original is None or image_original.ndim != 2:
            raise ValueError("Se requiere una imagen 2D en escala de grises.")
        image_pyramid = asm_utils.create_image_pyramid(
            image_original, self.ml_app_model.num_levels
        )
        image_pyramid.reverse()
        num_actual_levels = len(image_pyramid)

        if initial_pose_params:
            R, t, s, b = initial_pose_params
        else:
            R, t, s, b = self._initialize_pose_and_shape(image_pyramid[0].shape)

        for level_idx in range(num_actual_levels):
            app_model_level = (self.ml_app_model.num_levels - 1) - (
                num_actual_levels - 1 - level_idx
            )
            if app_model_level < 0:
                app_model_level = 0
            image_level = image_pyramid[level_idx]

            if verbose:
                print(
                    f"\n--- Ajustando Nivel {level_idx} (Modelo Ap. {app_model_level}) ---"
                )

            num_iterations = self.params["iterations_per_level"][app_model_level]
            for it in range(num_iterations):
                current_shape_procrustes = self.shape_model.reconstruct_shape(b)
                current_shape_img = asm_utils.transform_points(
                    current_shape_procrustes, R, t, s
                )
                if debug_plot_fn:
                    debug_plot_fn(image_level, current_shape_img, app_model_level, it)

                normals_img = asm_utils.calculate_normals(
                    current_shape_img, self.params["contour_indices_ordered"]
                )
                new_positions_img = np.zeros_like(current_shape_img)

                for lm_idx in range(self.shape_model.num_landmarks):
                    point, normal = current_shape_img[lm_idx], normals_img[lm_idx]
                    if np.all(normal == 0):
                        new_positions_img[lm_idx] = point
                        continue
                    app_model_lm = self.ml_app_model.get_landmark_model(
                        app_model_level, lm_idx
                    )
                    if app_model_lm is None or not app_model_lm._is_trained:
                        new_positions_img[lm_idx] = point
                        continue
                    new_positions_img[lm_idx] = self._search_one_landmark(
                        image_level, point, normal, app_model_lm
                    )

                R_old, t_old, s_old, b_old = R.copy(), t.copy(), s, b.copy()
                R, t, s, b = self._update_pose_and_shape(new_positions_img, R, t, s, b)

                param_change = (
                    np.linalg.norm(b - b_old)
                    + np.linalg.norm(t - t_old)
                    + abs(s - s_old)
                )
                if param_change < self.params["convergence_threshold"] and it > 3:
                    if verbose:
                        print(
                            f"Convergido en Nivel {app_model_level} en iteración {it + 1}."
                        )
                    break

            if level_idx < num_actual_levels - 1:
                R, t, s, b = self._propagate_params_to_next_level(
                    R, t, s, b, image_level.shape, image_pyramid[level_idx + 1].shape
                )

        final_shape_procrustes = self.shape_model.reconstruct_shape(b)
        scale_x = image_original.shape[1] / image_pyramid[-1].shape[1]
        scale_y = image_original.shape[0] / image_pyramid[-1].shape[0]
        avg_scale_to_orig = (scale_x + scale_y) / 2.0
        s_final_orig = s * avg_scale_to_orig
        t_final_orig = t * avg_scale_to_orig
        final_shape_img_coords = asm_utils.transform_points(
            final_shape_procrustes, R, t_final_orig, s_final_orig
        )

        return final_shape_img_coords, b
