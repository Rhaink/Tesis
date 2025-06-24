import numpy as np
from sklearn.decomposition import PCA
import pickle
# Necesitaremos las funciones de GPA y otras de asm_utils
# Asegúrate de que la ruta de importación sea correcta según la estructura del proyecto
# Si 'core' y 'utils' son subdirectorios de 'src', y 'src' está en PYTHONPATH o se ejecuta desde 'src'
# from ..utils import asm_utils # Esto sería si se ejecuta como parte de un paquete
# Por ahora, para scripts, podría ser más directo si se ajusta sys.path o se usa una importación relativa diferente.
# Asumamos por ahora que asm_utils estará accesible.
# Para una estructura de paquete típica, si ejecutas scripts desde fuera de src:
# from src.utils import asm_utils (si src está en PYTHONPATH)
# o si el script está en src/scripts:
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Añadir src al path
# from utils import asm_utils
# Por simplicidad, usaré una importación que funcionaría si 'src' es la raíz para los módulos.
# Esto podría necesitar ajuste dependiendo de cómo se ejecuten los scripts.
from utils import asm_utils # Corregido para la ejecución desde scripts/ que añade src/ al path

class ShapeModel:
    """
    Representa el Modelo Estadístico de Forma (SSM).
    Combina el Análisis Procrustes Generalizado (GPA) y el Análisis de Componentes Principales (PCA)
    para modelar la variabilidad de un conjunto de formas de entrenamiento.
    """
    def __init__(self, num_landmarks=None):
        """
        Inicializa el ShapeModel.
        Args:
            num_landmarks (int, optional): Número de landmarks por forma. Puede inferirse de los datos.
        """
        self.num_landmarks = num_landmarks
        self.mean_shape_procrustes = None  # Forma media después de GPA (num_landmarks, 2)
        self.pca_model = None              # Objeto PCA de scikit-learn ajustado
        self._is_trained = False

    def train(self, shapes_list, pca_n_components=0.98):
        """
        Entrena el modelo de forma a partir de una lista de formas.
        Args:
            shapes_list (np.array): Array de formas de entrenamiento (num_formas, num_landmarks, 2).
            pca_n_components (int or float): Número de componentes para PCA o varianza a retener.
        """
        if shapes_list.ndim != 3 or shapes_list.shape[2] != 2:
            raise ValueError("shapes_list debe ser un array 3D con la última dimensión de tamaño 2 (x,y).")
        if shapes_list.shape[0] < 2:
            raise ValueError("Se necesitan al menos 2 formas para entrenar el modelo.")

        if self.num_landmarks is None:
            self.num_landmarks = shapes_list.shape[1]
        elif self.num_landmarks != shapes_list.shape[1]:
            raise ValueError(f"El número de landmarks en shapes_list ({shapes_list.shape[1]}) no coincide con el num_landmarks del modelo ({self.num_landmarks}).")

        print("Iniciando entrenamiento del Modelo de Forma...")
        # 1. Análisis Procrustes Generalizado (GPA)
        print("Realizando Análisis Procrustes Generalizado (GPA)...")
        aligned_shapes, mean_aligned_shape = asm_utils.generalized_procrustes_analysis(shapes_list.copy())
        
        if aligned_shapes.size == 0 or mean_aligned_shape.size == 0:
            print("ERROR: GPA no devolvió resultados válidos. El entrenamiento del modelo de forma falló.")
            self._is_trained = False
            return False
        
        self.mean_shape_procrustes = mean_aligned_shape
        print(f"GPA completado. Forma media Procrustes calculada con {self.mean_shape_procrustes.shape[0]} landmarks.")

        # 2. Análisis de Componentes Principales (PCA)
        print("Construyendo modelo PCA...")
        # aligned_shapes ya está en el formato (num_formas, num_landmarks, 2)
        # build_pca_model espera (num_formas, num_features) donde num_features = num_landmarks * 2
        pca, mean_shape_vector_pca = asm_utils.build_pca_model(aligned_shapes, n_components=pca_n_components)
        
        if pca is None or not hasattr(pca, 'components_') or pca.components_ is None:
            print("ERROR: PCA no pudo construir un modelo válido. El entrenamiento del modelo de forma falló.")
            self.mean_shape_procrustes = None # Invalidar también la media de GPA si PCA falla
            self._is_trained = False
            return False

        self.pca_model = pca
        # La media de PCA (pca.mean_) debe ser muy similar a mean_aligned_shape aplanada,
        # ya que PCA se ajustó sobre las formas alineadas cuya media es mean_aligned_shape.
        # Podríamos verificar esto: np.allclose(pca.mean_, mean_aligned_shape.flatten())
        
        print(f"Modelo PCA construido. {self.pca_model.n_components_} componentes principales retenidos.")
        explained_variance_ratio = self.pca_model.explained_variance_ratio_
        if explained_variance_ratio is not None:
            print(f"Varianza total explicada: {np.sum(explained_variance_ratio)*100:.2f}%")

        self._is_trained = True
        print("Entrenamiento del Modelo de Forma completado exitosamente.")
        return True

    def reconstruct_shape(self, b_params):
        """
        Reconstruye una forma en el espacio Procrustes a partir de los parámetros de forma b.
        Forma = Media_Procrustes + P @ b
        Args:
            b_params (np.array): Vector de parámetros de forma (pesos para los componentes principales).
                                 Debe tener longitud pca_model.n_components_.
        Returns:
            np.array: Forma reconstruida (num_landmarks, 2) en el espacio Procrustes.
                      Devuelve None si el modelo no está entrenado o b_params es incorrecto.
        """
        if not self._is_trained:
            print("Error: El modelo de forma no ha sido entrenado.")
            return None
        if self.pca_model is None or self.mean_shape_procrustes is None:
            print("Error: Componentes del modelo PCA o forma media no disponibles.")
            return None
        
        # b_params debe ser un vector 1D
        b_params = np.asarray(b_params).flatten()

        if b_params.shape[0] != self.pca_model.n_components_:
            raise ValueError(f"La longitud de b_params ({b_params.shape[0]}) debe coincidir con el número de componentes del modelo PCA ({self.pca_model.n_components_}).")

        # pca.transform() espera (n_samples, n_features) y devuelve (n_samples, n_components)
        # pca.inverse_transform() espera (n_samples, n_components) y devuelve (n_samples, n_features)
        # La forma media de PCA (self.pca_model.mean_) ya está incorporada en inverse_transform.
        
        # b_params es el vector de pesos en el espacio de componentes.
        # Lo que pca.inverse_transform(b_params.reshape(1,-1)) hace es:
        #   reconstructed_flat = self.pca_model.mean_ + b_params @ self.pca_model.components_
        reconstructed_flat_shape = self.pca_model.inverse_transform(b_params.reshape(1, -1))[0]
        
        # La forma reconstruida debe tener num_landmarks * 2 elementos.
        return reconstructed_flat_shape.reshape(self.num_landmarks, 2)

    def project_to_shape_space(self, shape_procrustes):
        """
        Proyecta una forma (que ya está en el espacio Procrustes, es decir, alineada y normalizada)
        al espacio de parámetros de forma 'b'.
        b = P.T @ (Forma_Procrustes - Media_Procrustes_PCA)
        Args:
            shape_procrustes (np.array): Forma (num_landmarks, 2) en el espacio Procrustes.
        Returns:
            np.array: Vector de parámetros de forma 'b'.
                      Devuelve None si el modelo no está entrenado.
        """
        if not self._is_trained:
            print("Error: El modelo de forma no ha sido entrenado.")
            return None
        if self.pca_model is None:
            print("Error: Modelo PCA no disponible.")
            return None

        if shape_procrustes.shape != (self.num_landmarks, 2):
            raise ValueError(f"La forma de entrada debe ser ({self.num_landmarks}, 2).")

        flat_shape = shape_procrustes.flatten().reshape(1, -1)
        
        # pca.transform() se encarga de restar la media y proyectar:
        # b_params = (flat_shape - self.pca_model.mean_) @ self.pca_model.components_.T
        b_params = self.pca_model.transform(flat_shape)[0]
        return b_params

    def get_mean_shape_procrustes(self):
        """Devuelve la forma media Procrustes."""
        if not self._is_trained:
            print("Advertencia: El modelo no está entrenado. Devolviendo None para la forma media.")
        return self.mean_shape_procrustes

    def get_principal_components(self):
        """Devuelve la matriz de componentes principales (P). Cada fila es un componente."""
        if not self._is_trained or self.pca_model is None:
            print("Advertencia: El modelo no está entrenado o PCA no disponible. Devolviendo None.")
            return None
        return self.pca_model.components_ # (n_components, n_features)

    def get_eigenvalues(self):
        """Devuelve los eigenvalores (varianza explicada por cada componente)."""
        if not self._is_trained or self.pca_model is None:
            print("Advertencia: El modelo no está entrenado o PCA no disponible. Devolviendo None.")
            return None
        return self.pca_model.explained_variance_

    def save(self, path):
        """Guarda el modelo de forma entrenado en un archivo."""
        if not self._is_trained:
            print("Error: No se puede guardar un modelo de forma no entrenado.")
            return False
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'num_landmarks': self.num_landmarks,
                    'mean_shape_procrustes': self.mean_shape_procrustes,
                    'pca_model': self.pca_model
                }, f)
            print(f"Modelo de Forma guardado en: {path}")
            return True
        except Exception as e:
            print(f"Error al guardar el Modelo de Forma en {path}: {e}")
            return False

    @classmethod
    def load(cls, path):
        """Carga un modelo de forma entrenado desde un archivo."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            model = cls(num_landmarks=data['num_landmarks'])
            model.mean_shape_procrustes = data['mean_shape_procrustes']
            model.pca_model = data['pca_model']
            model._is_trained = True # Asumir que si se carga, está entrenado
            
            # Verificaciones básicas
            if model.mean_shape_procrustes is None or model.pca_model is None:
                 print(f"Advertencia: Modelo cargado desde {path} parece incompleto (media o PCA es None).")
                 model._is_trained = False
            elif model.num_landmarks != model.mean_shape_procrustes.shape[0]:
                 print(f"Advertencia: Discrepancia de landmarks en modelo cargado desde {path}.")
                 model._is_trained = False

            print(f"Modelo de Forma cargado desde: {path}")
            return model
        except FileNotFoundError:
            print(f"Error: Archivo de Modelo de Forma no encontrado en {path}")
            return None
        except Exception as e:
            print(f"Error al cargar el Modelo de Forma desde {path}: {e}")
            return None

if __name__ == '__main__':
    # Ejemplo de uso (requiere que asm_utils.py esté en una ruta accesible)
    print("Ejemplo de uso de ShapeModel:")
    
    # Crear datos de formas de ejemplo (3 formas, 4 landmarks, 2D)
    example_shapes = np.array([
        [[0,0], [1,0], [1,1], [0,1]],      # Cuadrado
        [[0.1,0.1], [1.1,0.2], [0.9,1.1], [0.2,0.8]], # Cuadrado deformado 1
        [[-0.1,0], [0.8,-0.1], [1.2,0.9], [-0.2,1.1]]  # Cuadrado deformado 2
    ])
    print(f"Formas de ejemplo creadas con dimensiones: {example_shapes.shape}")

    # Crear e instanciar el modelo de forma
    shape_model = ShapeModel(num_landmarks=4)
    
    # Entrenar
    training_successful = shape_model.train(example_shapes, pca_n_components=2) # Pedir 2 componentes

    if training_successful:
        print("\n--- Información del Modelo Entrenado ---")
        print(f"Forma Media Procrustes (primeros 2 landmarks):\n{shape_model.get_mean_shape_procrustes()[:2,:]}")
        print(f"Número de componentes PCA: {shape_model.pca_model.n_components_}")
        print(f"Eigenvalores (varianza explicada por componente):\n{shape_model.get_eigenvalues()}")
        print(f"Componentes Principales (P) (forma: {shape_model.get_principal_components().shape}):\n{shape_model.get_principal_components()}")

        # Reconstruir la forma media (usando b=[0,0,...])
        b_zeros = np.zeros(shape_model.pca_model.n_components_)
        reconstructed_mean = shape_model.reconstruct_shape(b_zeros)
        if reconstructed_mean is not None:
            print(f"\nForma reconstruida con b={b_zeros} (debería ser la media Procrustes):\n{reconstructed_mean[:2,:]}")
            # Comprobar si es cercano a la media Procrustes
            if np.allclose(reconstructed_mean, shape_model.get_mean_shape_procrustes()):
                print("Reconstrucción de la media verificada.")
            else:
                print("Error: Reconstrucción de la media no coincide con la media Procrustes.")

        # Probar reconstrucción con el primer modo de variación
        if shape_model.pca_model.n_components_ > 0:
            b_mode1 = np.zeros(shape_model.pca_model.n_components_)
            # Mover a lo largo del primer modo por +1 desviación estándar (sqrt(eigenvalue))
            # Los parámetros 'b' suelen estar en unidades de sqrt(eigenvalue)
            # Si b_k = 1, significa 1 * sqrt(lambda_k) * P_k
            # Aquí, si queremos que b_params[0] sea 1, significa que el peso es 1.
            b_mode1[0] = 1.0 
            shape_mode1 = shape_model.reconstruct_shape(b_mode1)
            if shape_mode1 is not None:
                print(f"\nForma reconstruida con b={b_mode1} (1er modo):\n{shape_mode1[:2,:]}")

        # Probar proyección de una forma al espacio 'b'
        # Tomemos la primera forma alineada del entrenamiento (si GPA funcionó)
        # Esto requiere acceso a aligned_shapes desde el entrenamiento, lo cual no está almacenado en el modelo.
        # En su lugar, proyectemos la forma media Procrustes. Debería dar b=[0,0,...]
        mean_procrustes_for_proj = shape_model.get_mean_shape_procrustes()
        b_params_for_mean = shape_model.project_to_shape_space(mean_procrustes_for_proj)
        if b_params_for_mean is not None:
            print(f"\nParámetros 'b' para la forma media Procrustes (deberían ser cercanos a cero):\n{b_params_for_mean}")
            if np.allclose(b_params_for_mean, 0):
                print("Proyección de la forma media a 'b' verificada.")
            else:
                print("Error: Proyección de la forma media a 'b' no dio ceros.")
        
        # Guardar y cargar el modelo
        # Ajustado para la nueva estructura, asumiendo que 'models' está al mismo nivel que 'src'
        model_path = "../models/example_shape_model.pkl" # Asegúrate que la carpeta exista
        print(f"\nIntentando guardar modelo en: {model_path}")
        if shape_model.save(model_path):
            print("Cargando modelo...")
            loaded_model = ShapeModel.load(model_path)
            if loaded_model and loaded_model._is_trained:
                print("Modelo cargado exitosamente.")
                # Verificar que el modelo cargado funciona
                b_zeros_loaded = np.zeros(loaded_model.pca_model.n_components_)
                reconstructed_mean_loaded = loaded_model.reconstruct_shape(b_zeros_loaded)
                if reconstructed_mean_loaded is not None and \
                   np.allclose(reconstructed_mean_loaded, loaded_model.get_mean_shape_procrustes()):
                    print("Modelo cargado verificado (reconstrucción de media).")
    else:
        print("El entrenamiento del modelo de forma falló.")

    print("\nFin del ejemplo de ShapeModel.")
