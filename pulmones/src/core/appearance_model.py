import os
import pickle

import numpy as np
from utils import asm_utils


class AppearanceModel:
    def __init__(self, profile_length=None):
        self.profile_length = profile_length
        self.mean_profile_derivative = None
        self.covariance_matrix_profile = None
        self.inv_covariance_matrix = None
        self._is_trained = False

    def train(self, profiles_list):
        if profiles_list.ndim != 2:
            raise ValueError("profiles_list debe ser un array 2D.")
        if profiles_list.shape[0] < 2:
            if profiles_list.shape[0] < 1:
                self._is_trained = False
                return False
            self.mean_profile_derivative = profiles_list[0]
            self.profile_length = profiles_list.shape[1]
            self.covariance_matrix_profile = np.eye(self.profile_length) * 1e-9
            self.inv_covariance_matrix = np.eye(self.profile_length) / 1e-9
            self._is_trained = True
            return True

        if self.profile_length is None:
            self.profile_length = profiles_list.shape[1]
        elif self.profile_length != profiles_list.shape[1]:
            raise ValueError("Longitud de perfiles no coincide.")

        self.mean_profile_derivative = np.mean(profiles_list, axis=0)
        self.covariance_matrix_profile = np.cov(profiles_list, rowvar=False)
        self.covariance_matrix_profile += np.eye(self.profile_length) * 1e-6
        try:
            self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix_profile)
        except np.linalg.LinAlgError:
            self.inv_covariance_matrix = np.linalg.pinv(self.covariance_matrix_profile)
        self._is_trained = True
        return True

    def calculate_match_cost(self, profile_derivative):
        if not self._is_trained:
            return float("inf")
        if profile_derivative.shape[0] != self.profile_length:
            raise ValueError("Longitud de perfil de entrada no coincide.")
        diff = profile_derivative - self.mean_profile_derivative
        return diff.T @ self.inv_covariance_matrix @ diff


class MultiLevelAppearanceModel:
    def __init__(self, num_levels=None, num_landmarks=None, profile_params=None):
        self.num_levels = num_levels
        self.num_landmarks = num_landmarks
        self.profile_params = profile_params if profile_params else {}
        self.levels_models = []
        self._is_trained = False

    def train(
        self,
        images_list,
        shapes_list_orig_coords,
        num_levels,
        num_landmarks,
        profile_params,
        contour_indices_ordered,
    ):
        self.num_levels = num_levels
        self.num_landmarks = num_landmarks
        self.profile_params = profile_params
        self.levels_models = [
            [
                AppearanceModel(profile_length=profile_params.get("num_points"))
                for _ in range(num_landmarks)
            ]
            for _ in range(num_levels)
        ]
        all_profiles_data = [
            [[] for _ in range(num_landmarks)] for _ in range(num_levels)
        ]

        print("Iniciando entrenamiento de Modelos de Apariencia Multinivel...")
        print(
            f"Parámetros de perfil: Longitud={profile_params['length']}, Puntos={profile_params['num_points']}"
        )

        for img_idx, image_orig in enumerate(images_list):
            if image_orig is None:
                continue
            shape_orig_for_img = shapes_list_orig_coords[img_idx]
            image_pyramid = asm_utils.create_image_pyramid(image_orig, self.num_levels)

            for level in range(min(self.num_levels, len(image_pyramid))):
                image_level = image_pyramid[level]
                h_level, w_level = image_level.shape
                shape_level = shape_orig_for_img.copy().astype(float)
                ref_w, ref_h = 64.0, 64.0
                shape_level[:, 0] *= w_level / ref_w
                shape_level[:, 1] *= h_level / ref_h
                normals_level = asm_utils.calculate_normals(
                    shape_level, contour_indices_ordered
                )

                for lm_idx in range(self.num_landmarks):
                    if np.all(normals_level[lm_idx] == 0):
                        continue

                    point_lm_level = shape_level[lm_idx]
                    normal_lm_level = normals_level[lm_idx]

                    ### CAMBIO CLAVE AQUÍ ###
                    # asm_utils.get_profile ahora devuelve una tupla (profile, edge_strength).
                    # Para entrenar el modelo de apariencia, solo necesitamos el perfil.
                    profile_data = asm_utils.get_profile(
                        image_level,
                        point_lm_level,
                        normal_lm_level,
                        self.profile_params["length"],
                        self.profile_params["num_points"],
                    )

                    # Extraemos solo el primer elemento (el perfil de derivada)
                    profile_deriv = profile_data[0]

                    if profile_deriv is not None and not np.all(profile_deriv == 0):
                        all_profiles_data[level][lm_idx].append(profile_deriv)

            if (img_idx + 1) % 50 == 0:
                print(
                    f"Procesadas {img_idx + 1}/{len(images_list)} imágenes para perfiles de apariencia."
                )

        print("\nEntrenando modelos de apariencia individuales por nivel y landmark...")
        for level in range(self.num_levels):
            num_trained_lm_models_level = 0
            for lm_idx in range(self.num_landmarks):
                # El problema de 'setting an array element with a sequence' se resuelve antes de este punto
                profiles_for_lm_level = np.array(all_profiles_data[level][lm_idx])
                if profiles_for_lm_level.shape[0] > 0:
                    if self.levels_models[level][lm_idx].train(profiles_for_lm_level):
                        num_trained_lm_models_level += 1
                else:
                    print(
                        f"Advertencia: No hay perfiles para el landmark {lm_idx} en el nivel {level}."
                    )
            print(
                f"Nivel {level}: {num_trained_lm_models_level}/{self.num_landmarks} modelos de landmark entrenados."
            )

        self._is_trained = True
        print("Entrenamiento de Modelos de Apariencia Multinivel completado.")
        return True

    def get_landmark_model(self, level, landmark_idx):
        if not self._is_trained:
            return None
        if not (
            0 <= level < self.num_levels and 0 <= landmark_idx < self.num_landmarks
        ):
            raise IndexError("Índice de nivel o landmark fuera de rango.")
        return self.levels_models[level][landmark_idx]

    def save(self, base_path):
        if not self._is_trained:
            return False
        try:
            meta_path = f"{base_path}_meta.pkl"
            with open(meta_path, "wb") as f_meta:
                pickle.dump(
                    {
                        "num_levels": self.num_levels,
                        "num_landmarks": self.num_landmarks,
                        "profile_params": self.profile_params,
                        "level_model_filenames": [
                            f"{os.path.basename(base_path)}_level_{lvl}.pkl"
                            for lvl in range(self.num_levels)
                        ],
                    },
                    f_meta,
                )
            for level in range(self.num_levels):
                level_model_path = f"{base_path}_level_{level}.pkl"
                with open(level_model_path, "wb") as f_level:
                    pickle.dump(self.levels_models[level], f_level)
            return True
        except Exception as e:
            print(f"Error al guardar Modelos de Apariencia: {e}")
            return False

    @classmethod
    def load(cls, base_path):
        meta_path = f"{base_path}_meta.pkl"
        try:
            with open(meta_path, "rb") as f_meta:
                meta_data = pickle.load(f_meta)

            model = cls(
                num_levels=meta_data["num_levels"],
                num_landmarks=meta_data["num_landmarks"],
                profile_params=meta_data["profile_params"],
            )
            model.levels_models = []
            for level in range(model.num_levels):
                level_model_filename = meta_data["level_model_filenames"][level]
                level_model_path = os.path.join(
                    os.path.dirname(base_path), level_model_filename
                )
                with open(level_model_path, "rb") as f_level:
                    model.levels_models.append(pickle.load(f_level))
            model._is_trained = True
            print(f"Modelos de Apariencia Multinivel cargados desde base: {base_path}")
            return model
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error al cargar Modelos de Apariencia: {e}")
            return None
