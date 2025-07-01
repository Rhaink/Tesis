"""
Eigenpatches model for landmark detection using PCA-based template matching.
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
import pickle
from pathlib import Path


class EigenpatchesModel:
    """
    PCA-based eigenpatches model for landmark detection.
    Uses Principal Component Analysis to create a compact representation
    of image patches around landmarks for template matching.
    """
    
    def __init__(self, patch_size: int = 21, n_components: int = 20):
        """
        Initialize the eigenpatches model.
        
        Args:
            patch_size: Size of square patches to extract (default 21x21)
            n_components: Number of principal components to retain
        """
        self.patch_size = patch_size
        self.n_components = n_components
        self.pca_models = {}  # One PCA model per landmark
        self.mean_patches = {}  # Mean patch per landmark
        self.n_landmarks = 0
        self.is_trained = False
        
    def _extract_patch(self, image: np.ndarray, x: float, y: float) -> np.ndarray:
        """
        Extract a square patch from image centered at (x, y).
        
        Args:
            image: Input grayscale image
            x, y: Center coordinates of patch
            
        Returns:
            Extracted patch of size (patch_size, patch_size)
        """
        half_size = self.patch_size // 2
        h, w = image.shape
        
        # Calculate patch boundaries
        y1 = max(0, int(y - half_size))
        y2 = min(h, int(y + half_size + 1))
        x1 = max(0, int(x - half_size))
        x2 = min(w, int(x + half_size + 1))
        
        # Extract patch
        patch = image[y1:y2, x1:x2]
        
        # Pad if patch is smaller than desired size
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded_patch = np.zeros((self.patch_size, self.patch_size))
            pad_y = (self.patch_size - patch.shape[0]) // 2
            pad_x = (self.patch_size - patch.shape[1]) // 2
            padded_patch[pad_y:pad_y+patch.shape[0], pad_x:pad_x+patch.shape[1]] = patch
            patch = padded_patch
            
        return patch.astype(np.float32)
    
    def train(self, images: List[np.ndarray], landmarks_list: List[np.ndarray]):
        """
        Train the eigenpatches model on training data.
        
        Args:
            images: List of grayscale images
            landmarks_list: List of landmark arrays, each of shape (n_landmarks, 2)
        """
        if len(images) != len(landmarks_list):
            raise ValueError("Number of images must match number of landmark sets")
            
        if len(images) == 0:
            raise ValueError("No training data provided")
            
        # Determine number of landmarks
        self.n_landmarks = landmarks_list[0].shape[0]
        
        # Collect patches for each landmark
        landmark_patches = {i: [] for i in range(self.n_landmarks)}
        
        for img, landmarks in zip(images, landmarks_list):
            # Normalize image
            img_normalized = cv2.equalizeHist(img.astype(np.uint8)).astype(np.float32)
            
            for landmark_idx in range(self.n_landmarks):
                x, y = landmarks[landmark_idx]
                patch = self._extract_patch(img_normalized, x, y)
                landmark_patches[landmark_idx].append(patch.flatten())
        
        # Train PCA model for each landmark
        for landmark_idx in range(self.n_landmarks):
            patches = np.array(landmark_patches[landmark_idx])
            
            # Normalize patches (zero mean)
            mean_patch = np.mean(patches, axis=0)
            centered_patches = patches - mean_patch
            
            # Apply PCA
            pca = PCA(n_components=min(self.n_components, patches.shape[0] - 1))
            pca.fit(centered_patches)
            
            # Store models
            self.pca_models[landmark_idx] = pca
            self.mean_patches[landmark_idx] = mean_patch
        
        self.is_trained = True
        
    def _compute_patch_score(self, patch: np.ndarray, landmark_idx: int) -> float:
        """
        Compute similarity score between patch and landmark model.
        
        Args:
            patch: Input patch (flattened)
            landmark_idx: Index of landmark model to compare against
            
        Returns:
            Similarity score (higher is better)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before computing scores")
            
        pca = self.pca_models[landmark_idx]
        mean_patch = self.mean_patches[landmark_idx]
        
        # Center the patch
        centered_patch = patch - mean_patch
        
        # Project to PCA space and reconstruct
        projected = pca.transform(centered_patch.reshape(1, -1))
        reconstructed = pca.inverse_transform(projected).flatten()
        
        # Compute reconstruction error (negative score - lower error is better)
        error = np.sum((centered_patch - reconstructed) ** 2)
        return -error
    
    def predict_landmarks(self, image: np.ndarray, 
                         search_regions: Optional[List[Tuple[int, int, int, int]]] = None,
                         step_size: int = 2) -> np.ndarray:
        """
        Predict landmark positions using template matching.
        
        Args:
            image: Input grayscale image
            search_regions: List of (x1, y1, x2, y2) regions to search for each landmark.
                          If None, searches entire image for all landmarks.
            step_size: Step size for sliding window search
            
        Returns:
            Predicted landmarks of shape (n_landmarks, 2)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Normalize image
        img_normalized = cv2.equalizeHist(image.astype(np.uint8)).astype(np.float32)
        h, w = img_normalized.shape
        
        predicted_landmarks = np.zeros((self.n_landmarks, 2))
        
        # Default search region is entire image
        if search_regions is None:
            search_regions = [(0, 0, w, h)] * self.n_landmarks
        
        for landmark_idx in range(self.n_landmarks):
            x1, y1, x2, y2 = search_regions[landmark_idx]
            best_score = -np.inf
            best_position = (x1 + (x2-x1)//2, y1 + (y2-y1)//2)  # Default to center
            
            # Sliding window search
            for y in range(y1 + self.patch_size//2, y2 - self.patch_size//2, step_size):
                for x in range(x1 + self.patch_size//2, x2 - self.patch_size//2, step_size):
                    patch = self._extract_patch(img_normalized, x, y)
                    score = self._compute_patch_score(patch.flatten(), landmark_idx)
                    
                    if score > best_score:
                        best_score = score
                        best_position = (x, y)
            
            predicted_landmarks[landmark_idx] = best_position
        
        return predicted_landmarks
    
    def refine_landmarks(self, image: np.ndarray, initial_landmarks: np.ndarray,
                        search_radius: int = 10) -> np.ndarray:
        """
        Refine landmark positions using local search around initial estimates.
        
        Args:
            image: Input grayscale image
            initial_landmarks: Initial landmark estimates of shape (n_landmarks, 2)
            search_radius: Radius of local search region
            
        Returns:
            Refined landmarks of shape (n_landmarks, 2)
        """
        search_regions = []
        for x, y in initial_landmarks:
            x1 = max(0, int(x - search_radius))
            y1 = max(0, int(y - search_radius))
            x2 = min(image.shape[1], int(x + search_radius))
            y2 = min(image.shape[0], int(y + search_radius))
            search_regions.append((x1, y1, x2, y2))
        
        return self.predict_landmarks(image, search_regions, step_size=1)
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'patch_size': self.patch_size,
            'n_components': self.n_components,
            'n_landmarks': self.n_landmarks,
            'pca_models': self.pca_models,
            'mean_patches': self.mean_patches,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.patch_size = model_data['patch_size']
        self.n_components = model_data['n_components']
        self.n_landmarks = model_data['n_landmarks']
        self.pca_models = model_data['pca_models']
        self.mean_patches = model_data['mean_patches']
        self.is_trained = model_data['is_trained']


class MultiScaleEigenpatches:
    """
    Multi-scale eigenpatches model using image pyramids for coarse-to-fine search.
    """
    
    def __init__(self, patch_size: int = 21, n_components: int = 20, 
                 pyramid_levels: int = 3, scale_factor: float = 0.5):
        """
        Initialize multi-scale eigenpatches model.
        
        Args:
            patch_size: Size of patches at finest scale
            n_components: Number of PCA components
            pyramid_levels: Number of pyramid levels
            scale_factor: Scale factor between pyramid levels
        """
        self.patch_size = patch_size
        self.n_components = n_components
        self.pyramid_levels = pyramid_levels
        self.scale_factor = scale_factor
        self.models = {}  # One model per pyramid level
        self.is_trained = False
    
    def _build_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """Build image pyramid."""
        pyramid = [image]
        current = image
        
        for _ in range(self.pyramid_levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
            
        return pyramid[::-1]  # Coarsest to finest
    
    def train(self, images: List[np.ndarray], landmarks_list: List[np.ndarray]):
        """Train models at each pyramid level."""
        # Build pyramids for all training images
        pyramids = [self._build_pyramid(img) for img in images]
        
        # Train model at each level
        for level in range(self.pyramid_levels):
            level_images = [pyramid[level] for pyramid in pyramids]
            
            # Scale landmarks for this pyramid level
            scale = self.scale_factor ** (self.pyramid_levels - 1 - level)
            level_landmarks = [landmarks * scale for landmarks in landmarks_list]
            
            # Train eigenpatches model for this level
            model = EigenpatchesModel(self.patch_size, self.n_components)
            model.train(level_images, level_landmarks)
            self.models[level] = model
        
        self.is_trained = True
    
    def predict_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Predict landmarks using coarse-to-fine search."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        pyramid = self._build_pyramid(image)
        
        # Start with coarsest level
        landmarks = None
        search_radius = 20
        
        for level in range(self.pyramid_levels):
            model = self.models[level]
            
            if landmarks is None:
                # First level - search entire image
                landmarks = model.predict_landmarks(pyramid[level])
            else:
                # Refine from previous level
                scale_factor = 2.0  # Each level is 2x finer
                scaled_landmarks = landmarks * scale_factor
                landmarks = model.refine_landmarks(pyramid[level], scaled_landmarks, search_radius)
            
            # Reduce search radius for next level
            search_radius = max(5, search_radius // 2)
        
        return landmarks
    
    def refine_landmarks(self, image: np.ndarray, initial_landmarks: np.ndarray,
                        search_radius: int = 10) -> np.ndarray:
        """
        Refine landmark positions using the finest scale model.
        
        Args:
            image: Input grayscale image
            initial_landmarks: Initial landmark estimates
            search_radius: Radius of local search region
            
        Returns:
            Refined landmarks
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before refinement")
        
        # Use the finest scale model (last level)
        finest_model = self.models[self.pyramid_levels - 1]
        return finest_model.refine_landmarks(image, initial_landmarks, search_radius)
    
    def save_model(self, filepath: str):
        """Save multi-scale model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        # Save individual models
        filepath = Path(filepath)
        base_path = filepath.parent / filepath.stem
        
        for level, model in self.models.items():
            level_path = f"{base_path}_level_{level}.pkl"
            model.save_model(level_path)
        
        # Save meta information
        meta_data = {
            'patch_size': self.patch_size,
            'n_components': self.n_components,
            'pyramid_levels': self.pyramid_levels,
            'scale_factor': self.scale_factor,
            'is_trained': self.is_trained
        }
        
        with open(f"{base_path}_meta.pkl", 'wb') as f:
            pickle.dump(meta_data, f)
    
    def load_model(self, filepath: str):
        """Load multi-scale model."""
        filepath = Path(filepath)
        base_path = filepath.parent / filepath.stem
        
        # Load meta information
        with open(f"{base_path}_meta.pkl", 'rb') as f:
            meta_data = pickle.load(f)
        
        self.patch_size = meta_data['patch_size']
        self.n_components = meta_data['n_components']
        self.pyramid_levels = meta_data['pyramid_levels']
        self.scale_factor = meta_data['scale_factor']
        self.is_trained = meta_data['is_trained']
        
        # Load individual models
        self.models = {}
        for level in range(self.pyramid_levels):
            level_path = f"{base_path}_level_{level}.pkl"
            model = EigenpatchesModel(self.patch_size, self.n_components)
            model.load_model(level_path)
            self.models[level] = model