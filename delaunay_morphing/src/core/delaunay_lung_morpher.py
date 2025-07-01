#!/usr/bin/env python3
"""
Delaunay Triangulation and Morphing for Lung Shape Analysis

This module implements a comprehensive system for lung shape morphing using
Delaunay triangulation, similar to the ASM approach but with enhanced features
for medical image analysis.

Author: Medical Image Analysis System
Date: 2025
"""

import numpy as np
import cv2
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline, griddata
from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class LungMorphingResult:
    """Container for morphing results"""
    warped_image: np.ndarray
    triangulation: Delaunay
    source_landmarks: np.ndarray
    target_landmarks: np.ndarray
    affine_matrices: List[np.ndarray]
    morphing_params: Dict[str, Any]


class DelaunayLungMorpher:
    """
    Advanced Delaunay-based morphing system for lung shape analysis.
    
    This class implements triangulation-based morphing between lung shapes,
    supporting various interpolation methods and morphing strategies.
    """
    
    def __init__(self, num_landmarks: int = 15):
        """
        Initialize the Delaunay Lung Morpher.
        
        Args:
            num_landmarks: Number of landmarks per lung (default: 15)
        """
        self.num_landmarks = num_landmarks
        self.triangulation_cache = {}
        
        # Anatomically correct lung landmark connectivity
        self.contour_connections = [
            (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
            (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
        ]
        self.mediastinal_connections = [
            (0, 8), (8, 9), (9, 10), (10, 1)
        ]
        
    def create_triangulation(self, landmarks: np.ndarray, 
                           add_boundary: bool = True,
                           boundary_padding: float = 0.1) -> Delaunay:
        """
        Create Delaunay triangulation from landmarks with optional boundary points.
        
        Args:
            landmarks: Array of shape (n_points, 2) containing landmark coordinates
            add_boundary: Whether to add boundary points for better coverage
            boundary_padding: Padding factor for boundary points
            
        Returns:
            Delaunay triangulation object
        """
        points = landmarks.copy()
        
        if add_boundary:
            # Add boundary points for better triangulation coverage
            min_x, min_y = np.min(points, axis=0)
            max_x, max_y = np.max(points, axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            
            pad_x = width * boundary_padding
            pad_y = height * boundary_padding
            
            # Create boundary points
            boundary_points = np.array([
                [min_x - pad_x, min_y - pad_y],  # Top-left
                [max_x + pad_x, min_y - pad_y],  # Top-right
                [max_x + pad_x, max_y + pad_y],  # Bottom-right
                [min_x - pad_x, max_y + pad_y],  # Bottom-left
                [(min_x + max_x) / 2, min_y - pad_y],  # Top-center
                [(min_x + max_x) / 2, max_y + pad_y],  # Bottom-center
                [min_x - pad_x, (min_y + max_y) / 2],  # Left-center
                [max_x + pad_x, (min_y + max_y) / 2],  # Right-center
            ])
            
            points = np.vstack([points, boundary_points])
        
        return Delaunay(points)
    
    def compute_affine_transform(self, src_triangle: np.ndarray, 
                               dst_triangle: np.ndarray) -> np.ndarray:
        """
        Compute affine transformation matrix between two triangles.
        
        Args:
            src_triangle: Source triangle vertices (3, 2)
            dst_triangle: Destination triangle vertices (3, 2)
            
        Returns:
            3x3 affine transformation matrix
        """
        # Add homogeneous coordinates
        src_tri = np.float32(src_triangle)
        dst_tri = np.float32(dst_triangle)
        
        # Compute affine transform
        matrix = cv2.getAffineTransform(src_tri, dst_tri)
        
        # Convert to 3x3 homogeneous matrix
        full_matrix = np.eye(3)
        full_matrix[:2, :] = matrix
        
        return full_matrix
    
    def morph_image(self, image: np.ndarray,
                   source_landmarks: np.ndarray,
                   target_landmarks: np.ndarray,
                   output_shape: Optional[Tuple[int, int]] = None,
                   interpolation_method: str = 'bilinear',
                   alpha: float = 1.0) -> LungMorphingResult:
        """
        Perform image morphing using Delaunay triangulation.
        
        Args:
            image: Source image
            source_landmarks: Source landmark coordinates
            target_landmarks: Target landmark coordinates
            output_shape: Output image shape (height, width)
            interpolation_method: 'bilinear', 'cubic', or 'nearest'
            alpha: Morphing factor (0=source, 1=target, 0.5=halfway)
            
        Returns:
            LungMorphingResult containing warped image and metadata
        """
        if output_shape is None:
            output_shape = image.shape[:2]
            
        # Interpolate landmarks based on alpha
        interpolated_landmarks = (1 - alpha) * source_landmarks + alpha * target_landmarks
        
        # Create triangulations
        source_tri = self.create_triangulation(source_landmarks)
        target_tri = self.create_triangulation(interpolated_landmarks)
        
        # Initialize output image
        output_image = np.zeros((*output_shape, image.shape[2] if len(image.shape) > 2 else 1))
        
        # Store affine matrices for each triangle
        affine_matrices = []
        
        # Create interpolator for source image
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            
        interpolators = []
        for c in range(image.shape[2]):
            if interpolation_method == 'cubic':
                interp = RectBivariateSpline(
                    np.arange(image.shape[0]),
                    np.arange(image.shape[1]),
                    image[:, :, c],
                    kx=3, ky=3
                )
            else:
                interp = RectBivariateSpline(
                    np.arange(image.shape[0]),
                    np.arange(image.shape[1]),
                    image[:, :, c],
                    kx=1, ky=1
                )
            interpolators.append(interp)
        
        # Process each triangle
        for simplex in target_tri.simplices:
            # Skip triangles that include boundary points
            if np.any(simplex >= len(source_landmarks)):
                continue
                
            # Get triangle vertices
            src_tri_vertices = source_landmarks[simplex[:3]]
            dst_tri_vertices = interpolated_landmarks[simplex[:3]]
            
            # Compute affine transformation
            affine_matrix = self.compute_affine_transform(dst_tri_vertices, src_tri_vertices)
            affine_matrices.append(affine_matrix)
            
            # Find bounding box of destination triangle
            x_min, y_min = np.min(dst_tri_vertices, axis=0).astype(int)
            x_max, y_max = np.max(dst_tri_vertices, axis=0).astype(int)
            
            # Clip to output image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(output_shape[1] - 1, x_max)
            y_max = min(output_shape[0] - 1, y_max)
            
            # Create mesh grid for the bounding box
            x_coords, y_coords = np.meshgrid(
                np.arange(x_min, x_max + 1),
                np.arange(y_min, y_max + 1)
            )
            
            # Flatten coordinates
            points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
            
            # Check which points are inside the triangle
            def point_in_triangle(p, tri):
                """Check if point p is inside triangle tri using barycentric coordinates"""
                v0 = tri[2] - tri[0]
                v1 = tri[1] - tri[0]
                v2 = p - tri[0]
                
                dot00 = np.dot(v0, v0)
                dot01 = np.dot(v0, v1)
                dot02 = np.dot(v0, v2)
                dot11 = np.dot(v1, v1)
                dot12 = np.dot(v1, v2)
                
                inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom
                
                return (u >= 0) and (v >= 0) and (u + v <= 1)
            
            # Filter points inside triangle
            inside_mask = np.array([point_in_triangle(p, dst_tri_vertices) for p in points])
            inside_points = points[inside_mask]
            
            if len(inside_points) > 0:
                # Transform points back to source image
                homogeneous_points = np.column_stack([
                    inside_points,
                    np.ones(len(inside_points))
                ])
                source_points = homogeneous_points @ affine_matrix.T
                source_points = source_points[:, :2]
                
                # Sample from source image
                for c, interp in enumerate(interpolators):
                    values = interp(source_points[:, 1], source_points[:, 0], grid=False)
                    
                    # Assign values to output image
                    for idx, (x, y) in enumerate(inside_points):
                        if 0 <= y < output_shape[0] and 0 <= x < output_shape[1]:
                            output_image[y, x, c] = values[idx]
        
        # Remove extra dimension if grayscale
        if output_image.shape[2] == 1:
            output_image = output_image[:, :, 0]
            
        return LungMorphingResult(
            warped_image=output_image,
            triangulation=target_tri,
            source_landmarks=source_landmarks,
            target_landmarks=interpolated_landmarks,
            affine_matrices=affine_matrices,
            morphing_params={
                'alpha': alpha,
                'interpolation_method': interpolation_method,
                'output_shape': output_shape
            }
        )
    
    def create_morphing_sequence(self, image: np.ndarray,
                               source_landmarks: np.ndarray,
                               target_landmarks: np.ndarray,
                               num_frames: int = 10,
                               output_shape: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Create a sequence of morphed images for animation.
        
        Args:
            image: Source image
            source_landmarks: Source landmark coordinates
            target_landmarks: Target landmark coordinates
            num_frames: Number of frames in the sequence
            output_shape: Output image shape
            
        Returns:
            List of morphed images
        """
        frames = []
        
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            result = self.morph_image(
                image, 
                source_landmarks, 
                target_landmarks,
                output_shape=output_shape,
                alpha=alpha
            )
            frames.append(result.warped_image)
            
        return frames
    
    def visualize_triangulation(self, landmarks: np.ndarray,
                              image: Optional[np.ndarray] = None,
                              title: str = "Delaunay Triangulation",
                              show_anatomical_connections: bool = True) -> plt.Figure:
        """
        Visualize the Delaunay triangulation on landmarks.
        
        Args:
            landmarks: Landmark coordinates
            image: Optional background image
            title: Plot title
            show_anatomical_connections: Whether to show anatomical connections
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        if image is not None:
            ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            
        # Create triangulation
        tri = self.create_triangulation(landmarks, add_boundary=False)
        
        # Plot triangles
        for simplex in tri.simplices:
            triangle = landmarks[simplex]
            triangle = np.vstack([triangle, triangle[0]])  # Close the triangle
            ax.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.5, alpha=0.3)
            
        # Plot anatomical connections if requested
        if show_anatomical_connections:
            # Plot contour connections
            for i, j in self.contour_connections:
                ax.plot([landmarks[i, 0], landmarks[j, 0]], 
                       [landmarks[i, 1], landmarks[j, 1]], 
                       'g-', linewidth=2, alpha=0.8, label='Contour' if (i, j) == self.contour_connections[0] else '')
            
            # Plot mediastinal connections
            for i, j in self.mediastinal_connections:
                ax.plot([landmarks[i, 0], landmarks[j, 0]], 
                       [landmarks[i, 1], landmarks[j, 1]], 
                       'orange', linewidth=2, alpha=0.8, linestyle='--',
                       label='Mediastinal' if (i, j) == self.mediastinal_connections[0] else '')
            
        # Plot landmarks
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=50, zorder=5)
        
        # Annotate landmarks
        for i, (x, y) in enumerate(landmarks):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='yellow', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
        ax.set_title(title)
        ax.axis('equal')
        
        if image is not None:
            ax.set_xlim(0, image.shape[1])
            ax.set_ylim(image.shape[0], 0)
            
        if show_anatomical_connections:
            ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def compute_triangle_quality_metrics(self, triangulation: Delaunay) -> Dict[str, float]:
        """
        Compute quality metrics for the triangulation.
        
        Args:
            triangulation: Delaunay triangulation
            
        Returns:
            Dictionary of quality metrics
        """
        points = triangulation.points
        simplices = triangulation.simplices
        
        angles = []
        areas = []
        aspect_ratios = []
        
        for simplex in simplices:
            # Get triangle vertices
            tri = points[simplex]
            
            # Compute side lengths
            sides = np.array([
                np.linalg.norm(tri[1] - tri[0]),
                np.linalg.norm(tri[2] - tri[1]),
                np.linalg.norm(tri[0] - tri[2])
            ])
            
            # Compute area using cross product
            area = 0.5 * abs(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            areas.append(area)
            
            # Compute angles using law of cosines
            for i in range(3):
                a, b, c = sides[i], sides[(i+1)%3], sides[(i+2)%3]
                if a > 0 and b > 0:
                    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle) * 180 / np.pi
                    angles.append(angle)
            
            # Compute aspect ratio (longest side / shortest side)
            if np.min(sides) > 0:
                aspect_ratios.append(np.max(sides) / np.min(sides))
        
        return {
            'mean_angle': np.mean(angles),
            'min_angle': np.min(angles),
            'max_angle': np.max(angles),
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'mean_aspect_ratio': np.mean(aspect_ratios),
            'max_aspect_ratio': np.max(aspect_ratios)
        }
    
    def save_morphing_result(self, result: LungMorphingResult, filepath: Path):
        """Save morphing result to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
            
    def load_morphing_result(self, filepath: Path) -> LungMorphingResult:
        """Load morphing result from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class LungShapeMorphingAnalyzer:
    """
    Analyzer for lung shape morphing with medical imaging focus.
    Provides tools for analyzing morphological changes between lung states.
    """
    
    def __init__(self, morpher: DelaunayLungMorpher):
        """
        Initialize the analyzer.
        
        Args:
            morpher: DelaunayLungMorpher instance
        """
        self.morpher = morpher
        
    def compute_shape_difference(self, landmarks1: np.ndarray, 
                               landmarks2: np.ndarray) -> Dict[str, float]:
        """
        Compute various metrics for shape difference.
        
        Args:
            landmarks1: First set of landmarks
            landmarks2: Second set of landmarks
            
        Returns:
            Dictionary of shape difference metrics
        """
        # Euclidean distance per landmark
        distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
        
        # Procrustes distance (after alignment)
        landmarks1_centered = landmarks1 - np.mean(landmarks1, axis=0)
        landmarks2_centered = landmarks2 - np.mean(landmarks2, axis=0)
        
        # Compute optimal rotation
        H = landmarks1_centered.T @ landmarks2_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply rotation and compute aligned distance
        landmarks2_aligned = landmarks2_centered @ R.T
        procrustes_distances = np.linalg.norm(landmarks1_centered - landmarks2_aligned, axis=1)
        
        return {
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances),
            'procrustes_mean': np.mean(procrustes_distances),
            'procrustes_max': np.max(procrustes_distances),
            'total_displacement': np.sum(distances)
        }
    
    def analyze_deformation_field(self, source_landmarks: np.ndarray,
                                target_landmarks: np.ndarray,
                                image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Compute dense deformation field from sparse landmarks.
        
        Args:
            source_landmarks: Source landmark coordinates
            target_landmarks: Target landmark coordinates
            image_shape: Shape of the output deformation field
            
        Returns:
            Deformation field of shape (height, width, 2)
        """
        # Compute displacement vectors
        displacements = target_landmarks - source_landmarks
        
        # Create grid for interpolation
        y_coords, x_coords = np.mgrid[0:image_shape[0], 0:image_shape[1]]
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # Interpolate displacement field
        dx = griddata(source_landmarks, displacements[:, 0], points, method='cubic')
        dy = griddata(source_landmarks, displacements[:, 1], points, method='cubic')
        
        # Reshape to image dimensions
        dx = dx.reshape(image_shape)
        dy = dy.reshape(image_shape)
        
        # Handle NaN values (extrapolation)
        dx = np.nan_to_num(dx, nan=0.0)
        dy = np.nan_to_num(dy, nan=0.0)
        
        return np.dstack([dx, dy])
    
    def visualize_deformation_field(self, deformation_field: np.ndarray,
                                  downsample: int = 20) -> plt.Figure:
        """
        Visualize deformation field as quiver plot.
        
        Args:
            deformation_field: Deformation field (height, width, 2)
            downsample: Downsampling factor for visualization
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Downsample for visualization
        y_coords = np.arange(0, deformation_field.shape[0], downsample)
        x_coords = np.arange(0, deformation_field.shape[1], downsample)
        
        # Extract displacement vectors
        dx = deformation_field[::downsample, ::downsample, 0]
        dy = deformation_field[::downsample, ::downsample, 1]
        
        # Compute magnitude for coloring
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Create quiver plot
        quiver = ax.quiver(x_coords, y_coords, dx, -dy, magnitude,
                          scale_units='xy', scale=1, cmap='viridis')
        
        ax.set_title('Deformation Field')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label('Displacement Magnitude')
        
        plt.tight_layout()
        return fig