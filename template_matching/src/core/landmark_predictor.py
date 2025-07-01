"""
Template matching landmark predictor combining eigenpatches with geometric constraints.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
try:
    from .eigenpatches import EigenpatchesModel, MultiScaleEigenpatches
except ImportError:
    from eigenpatches import EigenpatchesModel, MultiScaleEigenpatches
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


class TemplateLandmarkPredictor:
    """
    Landmark predictor that combines eigenpatches template matching with geometric constraints.
    """
    
    def __init__(self, 
                 patch_size: int = 21,
                 n_components: int = 20,
                 use_multiscale: bool = True,
                 pyramid_levels: int = 3):
        """
        Initialize the template landmark predictor.
        
        Args:
            patch_size: Size of patches for eigenpatches
            n_components: Number of PCA components
            use_multiscale: Whether to use multi-scale approach
            pyramid_levels: Number of pyramid levels for multi-scale
        """
        self.patch_size = patch_size
        self.n_components = n_components
        self.use_multiscale = use_multiscale
        self.pyramid_levels = pyramid_levels
        
        if use_multiscale:
            self.eigenpatch_model = MultiScaleEigenpatches(
                patch_size, n_components, pyramid_levels
            )
        else:
            self.eigenpatch_model = EigenpatchesModel(patch_size, n_components)
        
        # Shape statistics for geometric constraints
        self.mean_shape = None
        self.shape_cov = None
        self.shape_modes = None
        self.shape_eigenvalues = None
        self.is_trained = False
    
    def _compute_shape_statistics(self, landmarks_list: List[np.ndarray]):
        """
        Compute shape statistics for geometric constraints.
        
        Args:
            landmarks_list: List of landmark arrays for training
        """
        # Align shapes using Procrustes analysis (simplified version)
        aligned_shapes = []
        
        for landmarks in landmarks_list:
            # Center the shape
            centered = landmarks - np.mean(landmarks, axis=0)
            
            # Normalize by scale
            scale = np.sqrt(np.sum(centered ** 2))
            if scale > 0:
                normalized = centered / scale
                aligned_shapes.append(normalized.flatten())
        
        aligned_shapes = np.array(aligned_shapes)
        
        # Compute mean shape and covariance
        self.mean_shape = np.mean(aligned_shapes, axis=0)
        self.shape_cov = np.cov(aligned_shapes.T)
        
        # Compute shape modes (PCA)
        eigenvalues, eigenvectors = np.linalg.eigh(self.shape_cov)
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvalues)[::-1]
        self.shape_eigenvalues = eigenvalues[idx]
        self.shape_modes = eigenvectors[:, idx]
    
    def train(self, images: List[np.ndarray], landmarks_list: List[np.ndarray]):
        """
        Train the landmark predictor.
        
        Args:
            images: List of training images
            landmarks_list: List of corresponding landmarks
        """
        if len(images) != len(landmarks_list):
            raise ValueError("Number of images must match number of landmark sets")
        
        # Train eigenpatches model
        self.eigenpatch_model.train(images, landmarks_list)
        
        # Compute shape statistics
        self._compute_shape_statistics(landmarks_list)
        
        self.is_trained = True
    
    def _apply_shape_constraints(self, landmarks: np.ndarray, 
                               lambda_shape: float = 0.1) -> np.ndarray:
        """
        Apply shape constraints to refine landmark positions.
        
        Args:
            landmarks: Input landmarks of shape (n_landmarks, 2)
            lambda_shape: Weight for shape constraint
            
        Returns:
            Constrained landmarks
        """
        if self.mean_shape is None:
            return landmarks
        
        # Center and normalize input shape
        centered = landmarks - np.mean(landmarks, axis=0)
        scale = np.sqrt(np.sum(centered ** 2))
        if scale > 0:
            normalized = centered / scale
        else:
            return landmarks
        
        shape_vector = normalized.flatten()
        
        # Project onto shape space
        diff = shape_vector - self.mean_shape
        
        # Constrain to valid shape variations
        n_modes = min(10, len(self.shape_eigenvalues))  # Use top 10 modes
        projected = np.zeros_like(diff)
        
        for i in range(n_modes):
            mode = self.shape_modes[:, i]
            eigenval = self.shape_eigenvalues[i]
            
            # Project onto this mode
            projection = np.dot(diff, mode)
            
            # Constrain projection (within 3 standard deviations)
            max_proj = 3 * np.sqrt(eigenval)
            projection = np.clip(projection, -max_proj, max_proj)
            
            # Add to projected shape
            projected += projection * mode
        
        # Reconstruct shape
        constrained_shape = self.mean_shape + lambda_shape * projected
        constrained_shape = constrained_shape.reshape(-1, 2)
        
        # Denormalize
        constrained_shape = constrained_shape * scale + np.mean(landmarks, axis=0)
        
        return constrained_shape
    
    def predict_landmarks(self, image: np.ndarray, 
                         initial_guess: Optional[np.ndarray] = None,
                         max_iterations: int = 5,
                         lambda_shape: float = 0.1) -> Dict[str, Any]:
        """
        Predict landmarks using template matching with geometric constraints.
        
        Args:
            image: Input grayscale image
            initial_guess: Initial landmark positions (optional)
            max_iterations: Maximum refinement iterations
            lambda_shape: Weight for shape constraints
            
        Returns:
            Dictionary containing:
            - 'landmarks': Final landmark positions
            - 'iterations': Number of iterations used
            - 'convergence': Whether algorithm converged
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Initial prediction using eigenpatches
        if initial_guess is None:
            landmarks = self.eigenpatch_model.predict_landmarks(image)
        else:
            landmarks = initial_guess.copy()
        
        # Iterative refinement
        for iteration in range(max_iterations):
            prev_landmarks = landmarks.copy()
            
            # Refine using template matching
            landmarks = self.eigenpatch_model.refine_landmarks(
                image, landmarks, search_radius=10
            )
            
            # Apply shape constraints
            landmarks = self._apply_shape_constraints(landmarks, lambda_shape)
            
            # Check convergence
            displacement = np.mean(np.linalg.norm(landmarks - prev_landmarks, axis=1))
            if displacement < 0.5:  # Convergence threshold
                return {
                    'landmarks': landmarks,
                    'iterations': iteration + 1,
                    'convergence': True
                }
        
        return {
            'landmarks': landmarks,
            'iterations': max_iterations,
            'convergence': False
        }
    
    def predict_with_confidence(self, image: np.ndarray,
                              initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Predict landmarks with confidence scores.
        
        Args:
            image: Input grayscale image
            initial_guess: Initial landmark positions
            
        Returns:
            Dictionary containing landmarks, confidence scores, and metadata
        """
        result = self.predict_landmarks(image, initial_guess)
        landmarks = result['landmarks']
        
        # Compute confidence scores based on template matching quality
        confidence_scores = self._compute_confidence_scores(image, landmarks)
        
        result['confidence_scores'] = confidence_scores
        result['mean_confidence'] = np.mean(confidence_scores)
        
        return result
    
    def _compute_confidence_scores(self, image: np.ndarray, 
                                 landmarks: np.ndarray) -> np.ndarray:
        """
        Compute confidence scores for each landmark based on template matching quality.
        
        Args:
            image: Input image
            landmarks: Predicted landmarks
            
        Returns:
            Confidence scores for each landmark
        """
        img_normalized = cv2.equalizeHist(image.astype(np.uint8)).astype(np.float32)
        confidence_scores = np.zeros(len(landmarks))
        
        for i, (x, y) in enumerate(landmarks):
            # Extract patch at predicted location
            patch = self.eigenpatch_model._extract_patch(img_normalized, x, y)
            
            # Compute template matching score
            if hasattr(self.eigenpatch_model, 'models'):
                # Multi-scale model
                model = self.eigenpatch_model.models[0]  # Use finest scale
            else:
                # Single-scale model
                model = self.eigenpatch_model
            
            score = model._compute_patch_score(patch.flatten(), i)
            
            # Normalize score to [0, 1] range (heuristic)
            confidence_scores[i] = 1.0 / (1.0 + np.exp(-score / 1000))
        
        return confidence_scores
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save eigenpatches model
        self.eigenpatch_model.save_model(filepath)
        
        # Save shape statistics
        import pickle
        shape_data = {
            'mean_shape': self.mean_shape,
            'shape_cov': self.shape_cov,
            'shape_modes': self.shape_modes,
            'shape_eigenvalues': self.shape_eigenvalues,
            'patch_size': self.patch_size,
            'n_components': self.n_components,
            'use_multiscale': self.use_multiscale,
            'pyramid_levels': self.pyramid_levels
        }
        
        shape_filepath = filepath.replace('.pkl', '_shape.pkl')
        with open(shape_filepath, 'wb') as f:
            pickle.dump(shape_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        # Load eigenpatches model
        self.eigenpatch_model.load_model(filepath)
        
        # Load shape statistics
        import pickle
        shape_filepath = filepath.replace('.pkl', '_shape.pkl')
        with open(shape_filepath, 'rb') as f:
            shape_data = pickle.load(f)
        
        self.mean_shape = shape_data['mean_shape']
        self.shape_cov = shape_data['shape_cov']
        self.shape_modes = shape_data['shape_modes']
        self.shape_eigenvalues = shape_data['shape_eigenvalues']
        self.patch_size = shape_data['patch_size']
        self.n_components = shape_data['n_components']
        self.use_multiscale = shape_data['use_multiscale']
        self.pyramid_levels = shape_data['pyramid_levels']
        
        self.is_trained = True


class EnsembleLandmarkPredictor:
    """
    Ensemble of multiple template matching predictors for improved robustness.
    """
    
    def __init__(self, predictors: List[TemplateLandmarkPredictor]):
        """
        Initialize ensemble predictor.
        
        Args:
            predictors: List of trained landmark predictors
        """
        self.predictors = predictors
        self.n_predictors = len(predictors)
        
        if self.n_predictors == 0:
            raise ValueError("At least one predictor is required")
    
    def predict_landmarks(self, image: np.ndarray,
                         initial_guess: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Predict landmarks using ensemble of predictors.
        
        Args:
            image: Input image
            initial_guess: Initial landmark positions
            
        Returns:
            Dictionary with ensemble predictions and individual results
        """
        individual_results = []
        
        # Get predictions from each predictor
        for predictor in self.predictors:
            result = predictor.predict_with_confidence(image, initial_guess)
            individual_results.append(result)
        
        # Combine predictions using weighted average based on confidence
        all_landmarks = np.array([result['landmarks'] for result in individual_results])
        all_confidences = np.array([result['mean_confidence'] for result in individual_results])
        
        # Normalize confidence weights
        weights = all_confidences / np.sum(all_confidences)
        
        # Weighted average of landmarks
        ensemble_landmarks = np.average(all_landmarks, axis=0, weights=weights)
        
        # Compute ensemble confidence
        ensemble_confidence = np.mean(all_confidences)
        
        return {
            'landmarks': ensemble_landmarks,
            'confidence': ensemble_confidence,
            'individual_results': individual_results,
            'weights': weights
        }