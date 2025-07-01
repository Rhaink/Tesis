#!/usr/bin/env python3
"""
Advanced morphing analysis integrating Delaunay triangulation with ASM results.

This script provides advanced analysis capabilities by combining the Delaunay
morphing approach with the existing ASM framework for comprehensive lung
shape analysis.
"""

import sys
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
from scipy.stats import ttest_ind, f_oneway
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Add project paths
PROJECT_ROOT = Path("/home/donrobot/Projects/Tesis")
sys.path.append(str(PROJECT_ROOT / "delaunay_morphing/src"))
sys.path.append(str(PROJECT_ROOT / "pulmones/src"))

from core.delaunay_lung_morpher import DelaunayLungMorpher, LungShapeMorphingAnalyzer
from core.shape_model import ShapeModel
from core.appearance_model import MultiLevelAppearanceModel
from core.asm_fitter import ASMFitter


class AdvancedLungMorphingAnalyzer:
    """
    Advanced analyzer combining Delaunay morphing with ASM for comprehensive
    lung shape analysis in medical imaging.
    """
    
    def __init__(self, morpher: DelaunayLungMorpher):
        """Initialize the advanced analyzer."""
        self.morpher = morpher
        self.base_analyzer = LungShapeMorphingAnalyzer(morpher)
        self.shape_model = None
        self.appearance_model = None
        self.asm_fitter = None
        
    def load_asm_models(self, shape_model_path: Path, appearance_model_path: Path):
        """Load pre-trained ASM models."""
        print(f"Loading ASM models...")
        
        # Load shape model
        with open(shape_model_path, 'rb') as f:
            self.shape_model = pickle.load(f)
            
        # Load appearance model
        with open(appearance_model_path, 'rb') as f:
            self.appearance_model = pickle.load(f)
            
        # Create ASM fitter
        self.asm_fitter = ASMFitter(self.shape_model, self.appearance_model)
        print("✓ ASM models loaded successfully")
        
    def extract_shape_features(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive shape features from landmarks.
        
        Returns:
            Dictionary containing various shape descriptors
        """
        features = {}
        
        # Basic geometric features
        centroid = np.mean(landmarks, axis=0)
        centered = landmarks - centroid
        
        # Distances from centroid
        distances = np.linalg.norm(centered, axis=1)
        features['centroid_distances'] = distances
        
        # Angles from centroid
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        features['centroid_angles'] = angles
        
        # Shape compactness (perimeter^2 / area)
        # Using convex hull for area calculation
        from scipy.spatial import ConvexHull
        hull = ConvexHull(landmarks)
        area = hull.volume  # In 2D, volume is area
        perimeter = hull.area  # In 2D, area is perimeter
        features['compactness'] = perimeter**2 / (4 * np.pi * area)
        
        # Eccentricity (from PCA)
        pca = PCA(n_components=2)
        pca.fit(centered)
        eigenvalues = pca.explained_variance_
        features['eccentricity'] = np.sqrt(1 - eigenvalues[1] / eigenvalues[0])
        
        # Curvature at each landmark
        curvatures = []
        n_landmarks = len(landmarks)
        for i in range(n_landmarks):
            # Get three consecutive points
            p1 = landmarks[(i - 1) % n_landmarks]
            p2 = landmarks[i]
            p3 = landmarks[(i + 1) % n_landmarks]
            
            # Compute curvature using Menger curvature formula
            area = 0.5 * abs(np.cross(p3 - p1, p2 - p1))
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            if a * b * c > 0:
                curvature = 4 * area / (a * b * c)
            else:
                curvature = 0
            curvatures.append(curvature)
            
        features['curvatures'] = np.array(curvatures)
        
        # Fourier descriptors
        complex_coords = landmarks[:, 0] + 1j * landmarks[:, 1]
        fft = np.fft.fft(complex_coords)
        features['fourier_descriptors'] = np.abs(fft[:8])  # First 8 descriptors
        
        return features
    
    def analyze_morphing_trajectory(self, source_landmarks: np.ndarray,
                                  target_landmarks: np.ndarray,
                                  num_steps: int = 10) -> Dict[str, List]:
        """
        Analyze the morphing trajectory between two shapes.
        
        Returns:
            Dictionary containing trajectory analysis results
        """
        trajectory_data = {
            'alphas': [],
            'shape_features': [],
            'deformation_energy': [],
            'landmark_velocities': []
        }
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            trajectory_data['alphas'].append(alpha)
            
            # Interpolate landmarks
            current_landmarks = (1 - alpha) * source_landmarks + alpha * target_landmarks
            
            # Extract shape features
            features = self.extract_shape_features(current_landmarks)
            trajectory_data['shape_features'].append(features)
            
            # Compute deformation energy (elastic energy approximation)
            if i > 0:
                prev_landmarks = trajectory_data['shape_features'][i-1]
                displacement = current_landmarks - prev_landmarks
                energy = np.sum(displacement**2)
                trajectory_data['deformation_energy'].append(energy)
                
                # Compute landmark velocities
                velocities = np.linalg.norm(displacement, axis=1)
                trajectory_data['landmark_velocities'].append(velocities)
            else:
                trajectory_data['deformation_energy'].append(0)
                trajectory_data['landmark_velocities'].append(np.zeros(len(source_landmarks)))
                
        return trajectory_data
    
    def compare_morphing_methods(self, image: np.ndarray,
                               source_landmarks: np.ndarray,
                               target_landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compare different morphing methods: Delaunay vs TPS vs Linear.
        
        Returns:
            Dictionary with morphed images from different methods
        """
        results = {}
        
        # Delaunay morphing
        delaunay_result = self.morpher.morph_image(
            image, source_landmarks, target_landmarks, alpha=0.5
        )
        results['delaunay'] = delaunay_result.warped_image
        
        # Linear interpolation (simple blending)
        interpolated_landmarks = 0.5 * source_landmarks + 0.5 * target_landmarks
        linear_result = self.morpher.morph_image(
            image, source_landmarks, interpolated_landmarks, alpha=1.0
        )
        results['linear'] = linear_result.warped_image
        
        # Thin Plate Spline (TPS) morphing
        try:
            from cv2 import createThinPlateSplineShapeTransformer
            
            # Prepare matches
            matches = []
            for i in range(len(source_landmarks)):
                matches.append(cv2.DMatch(i, i, 0))
            
            # Create TPS transformer
            tps = createThinPlateSplineShapeTransformer()
            
            # Reshape for OpenCV
            src_pts = source_landmarks.reshape(1, -1, 2).astype(np.float32)
            dst_pts = (0.5 * source_landmarks + 0.5 * target_landmarks).reshape(1, -1, 2).astype(np.float32)
            
            # Estimate transformation
            tps.estimateTransformation(dst_pts, src_pts, matches)
            
            # Apply transformation
            tps_result = tps.warpImage(image)
            results['tps'] = tps_result
        except:
            print("TPS morphing not available (OpenCV contrib required)")
            results['tps'] = None
            
        return results
    
    def analyze_pathology_morphing_patterns(self, dataset_path: Path) -> Dict:
        """
        Analyze morphing patterns between different pathologies.
        
        Returns:
            Dictionary containing statistical analysis of morphing patterns
        """
        # Load dataset
        df = pd.read_csv(dataset_path, header=None)
        
        # Group by pathology
        pathology_shapes = {
            'COVID': [],
            'Normal': [],
            'Viral Pneumonia': []
        }
        
        for _, row in df.iterrows():
            image_name = row.iloc[-1]
            coords = row.iloc[:30].values.astype(float).reshape(15, 2)
            
            # Scale to image coordinates
            coords[:, 0] *= 299 / 64
            coords[:, 1] *= 299 / 64
            
            for pathology in pathology_shapes.keys():
                if pathology.replace(' ', '-') in image_name:
                    pathology_shapes[pathology].append(coords)
                    break
        
        # Compute mean shapes
        mean_shapes = {}
        for pathology, shapes in pathology_shapes.items():
            if shapes:
                mean_shapes[pathology] = np.mean(shapes, axis=0)
        
        # Analyze inter-pathology morphing
        morphing_analysis = {}
        
        for p1 in mean_shapes:
            for p2 in mean_shapes:
                if p1 != p2:
                    key = f"{p1}_to_{p2}"
                    
                    # Analyze morphing trajectory
                    trajectory = self.analyze_morphing_trajectory(
                        mean_shapes[p1], 
                        mean_shapes[p2],
                        num_steps=20
                    )
                    
                    # Compute statistics
                    total_energy = np.sum(trajectory['deformation_energy'])
                    max_velocity = np.max([np.max(v) for v in trajectory['landmark_velocities']])
                    
                    morphing_analysis[key] = {
                        'total_deformation_energy': total_energy,
                        'max_landmark_velocity': max_velocity,
                        'trajectory': trajectory
                    }
        
        # Statistical significance testing
        significance_tests = {}
        
        for p1, p2 in [('COVID', 'Normal'), ('COVID', 'Viral Pneumonia'), ('Normal', 'Viral Pneumonia')]:
            shapes1 = np.array(pathology_shapes[p1])
            shapes2 = np.array(pathology_shapes[p2])
            
            # Flatten shapes for t-test
            flat1 = shapes1.reshape(shapes1.shape[0], -1)
            flat2 = shapes2.reshape(shapes2.shape[0], -1)
            
            # Perform t-tests for each coordinate
            p_values = []
            for i in range(flat1.shape[1]):
                _, p_value = ttest_ind(flat1[:, i], flat2[:, i])
                p_values.append(p_value)
            
            significance_tests[f"{p1}_vs_{p2}"] = {
                'mean_p_value': np.mean(p_values),
                'significant_coords': np.sum(np.array(p_values) < 0.05),
                'total_coords': len(p_values)
            }
        
        return {
            'mean_shapes': mean_shapes,
            'morphing_analysis': morphing_analysis,
            'significance_tests': significance_tests,
            'sample_sizes': {k: len(v) for k, v in pathology_shapes.items()}
        }
    
    def create_morphing_atlas(self, dataset_path: Path, 
                            output_dir: Path,
                            num_exemplars: int = 5) -> None:
        """
        Create an atlas of morphing examples for each pathology transition.
        """
        print("Creating morphing atlas...")
        output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        df = pd.read_csv(dataset_path, header=None)
        
        # Group by pathology
        pathology_data = {
            'COVID': {'images': [], 'landmarks': []},
            'Normal': {'images': [], 'landmarks': []},
            'Viral Pneumonia': {'images': [], 'landmarks': []}
        }
        
        for _, row in df.iterrows():
            image_name = row.iloc[-1]
            coords = row.iloc[:30].values.astype(float).reshape(15, 2)
            
            # Scale coordinates
            coords[:, 0] *= 299 / 64
            coords[:, 1] *= 299 / 64
            
            # Load image
            image = None
            for pathology in pathology_data.keys():
                if pathology.replace(' ', '-') in image_name:
                    image_path = PROJECT_ROOT / f"COVID-19_Radiography_Dataset/{pathology}/images/{image_name}"
                    if image_path.exists():
                        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                        pathology_data[pathology]['images'].append(image)
                        pathology_data[pathology]['landmarks'].append(coords)
                    break
        
        # Create morphing examples for each pathology pair
        for p1 in pathology_data:
            for p2 in pathology_data:
                if p1 < p2 and pathology_data[p1]['images'] and pathology_data[p2]['images']:
                    print(f"\nCreating {p1} → {p2} morphing examples...")
                    
                    # Select random exemplars
                    n1 = min(num_exemplars, len(pathology_data[p1]['images']))
                    n2 = min(num_exemplars, len(pathology_data[p2]['images']))
                    
                    indices1 = np.random.choice(len(pathology_data[p1]['images']), n1, replace=False)
                    indices2 = np.random.choice(len(pathology_data[p2]['images']), n2, replace=False)
                    
                    # Create morphing grid
                    fig, axes = plt.subplots(n1, 5, figsize=(20, 4*n1))
                    if n1 == 1:
                        axes = axes.reshape(1, -1)
                    
                    for i, idx1 in enumerate(indices1):
                        # Select a random target
                        idx2 = np.random.choice(indices2)
                        
                        image1 = pathology_data[p1]['images'][idx1]
                        landmarks1 = pathology_data[p1]['landmarks'][idx1]
                        landmarks2 = pathology_data[p2]['landmarks'][idx2]
                        
                        # Create morphing sequence
                        alphas = [0, 0.25, 0.5, 0.75, 1.0]
                        
                        for j, alpha in enumerate(alphas):
                            result = self.morpher.morph_image(
                                image1, landmarks1, landmarks2, alpha=alpha
                            )
                            
                            axes[i, j].imshow(result.warped_image, cmap='gray')
                            axes[i, j].set_title(f'α = {alpha}')
                            axes[i, j].axis('off')
                            
                            # Add landmarks overlay for first and last
                            if j == 0:
                                axes[i, j].scatter(landmarks1[:, 0], landmarks1[:, 1], 
                                                 c='red', s=10, alpha=0.5)
                            elif j == 4:
                                interp_landmarks = result.target_landmarks
                                axes[i, j].scatter(interp_landmarks[:, 0], interp_landmarks[:, 1], 
                                                 c='blue', s=10, alpha=0.5)
                    
                    plt.suptitle(f'Morphing Examples: {p1} → {p2}', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(output_dir / f'morphing_atlas_{p1}_to_{p2}.png', dpi=150, bbox_inches='tight')
                    plt.close()
        
        print(f"\n✓ Morphing atlas saved to {output_dir}")
    
    def visualize_shape_space_morphing(self, dataset_path: Path) -> plt.Figure:
        """
        Visualize morphing trajectories in shape space using PCA/t-SNE.
        """
        # Load all shapes
        df = pd.read_csv(dataset_path, header=None)
        all_shapes = []
        labels = []
        
        for _, row in df.iterrows():
            image_name = row.iloc[-1]
            coords = row.iloc[:30].values.astype(float)
            all_shapes.append(coords)
            
            # Determine label
            if 'COVID' in image_name:
                labels.append('COVID')
            elif 'Normal' in image_name:
                labels.append('Normal')
            else:
                labels.append('Viral Pneumonia')
        
        all_shapes = np.array(all_shapes)
        
        # Apply PCA
        pca = PCA(n_components=2)
        shapes_pca = pca.fit_transform(all_shapes)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA visualization
        colors = {'COVID': 'red', 'Normal': 'green', 'Viral Pneumonia': 'blue'}
        for label in set(labels):
            mask = np.array(labels) == label
            ax1.scatter(shapes_pca[mask, 0], shapes_pca[mask, 1], 
                       c=colors[label], label=label, alpha=0.6, s=50)
        
        # Add morphing trajectories
        # Get mean shapes in PCA space
        mean_shapes_pca = {}
        for label in set(labels):
            mask = np.array(labels) == label
            mean_shapes_pca[label] = np.mean(shapes_pca[mask], axis=0)
        
        # Draw morphing paths
        for p1, p2 in [('COVID', 'Normal'), ('COVID', 'Viral Pneumonia'), ('Normal', 'Viral Pneumonia')]:
            start = mean_shapes_pca[p1]
            end = mean_shapes_pca[p2]
            
            # Create trajectory
            trajectory = np.array([start + alpha * (end - start) for alpha in np.linspace(0, 1, 20)])
            ax1.plot(trajectory[:, 0], trajectory[:, 1], 'k--', alpha=0.5, linewidth=2)
            
            # Add arrow
            ax1.annotate('', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='->', lw=2, alpha=0.7))
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('Shape Space (PCA) with Morphing Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # t-SNE visualization
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        shapes_tsne = tsne.fit_transform(all_shapes)
        
        for label in set(labels):
            mask = np.array(labels) == label
            ax2.scatter(shapes_tsne[mask, 0], shapes_tsne[mask, 1], 
                       c=colors[label], label=label, alpha=0.6, s=50)
        
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.set_title('Shape Space (t-SNE)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Lung Shape Space Visualization', fontsize=16)
        plt.tight_layout()
        
        return fig


def main():
    """Run advanced morphing analysis."""
    print("Advanced Lung Morphing Analysis")
    print("===============================\n")
    
    # Initialize analyzer
    morpher = DelaunayLungMorpher()
    analyzer = AdvancedLungMorphingAnalyzer(morpher)
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'delaunay_morphing/advanced_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Load ASM models if available
    shape_model_path = PROJECT_ROOT / "pulmones/models/shape_model_balanced_500_augmented.pkl"
    appearance_model_path = PROJECT_ROOT / "pulmones/models/appearance_model_balanced_500_augmented_meta.pkl"
    
    if shape_model_path.exists() and appearance_model_path.exists():
        analyzer.load_asm_models(shape_model_path, appearance_model_path)
    
    # Analyze pathology morphing patterns
    print("\n1. Analyzing pathology morphing patterns...")
    dataset_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    pathology_analysis = analyzer.analyze_pathology_morphing_patterns(dataset_path)
    
    print("\nSample sizes:")
    for pathology, count in pathology_analysis['sample_sizes'].items():
        print(f"  {pathology}: {count} samples")
    
    print("\nStatistical significance tests:")
    for comparison, results in pathology_analysis['significance_tests'].items():
        print(f"\n{comparison}:")
        print(f"  Mean p-value: {results['mean_p_value']:.4f}")
        print(f"  Significant coordinates: {results['significant_coords']}/{results['total_coords']}")
    
    # Create morphing atlas
    print("\n2. Creating morphing atlas...")
    analyzer.create_morphing_atlas(dataset_path, output_dir / 'atlas', num_exemplars=3)
    
    # Visualize shape space
    print("\n3. Visualizing shape space with morphing trajectories...")
    fig = analyzer.visualize_shape_space_morphing(dataset_path)
    fig.savefig(output_dir / 'shape_space_morphing.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare morphing methods
    print("\n4. Comparing morphing methods...")
    # Load a sample image
    df = pd.read_csv(dataset_path, header=None)
    sample_row = df.iloc[0]
    image_name = sample_row.iloc[-1]
    
    # Load image and landmarks
    image = None
    for category in ['COVID', 'Normal', 'Viral Pneumonia']:
        image_path = PROJECT_ROOT / f"COVID-19_Radiography_Dataset/{category}/images/{image_name}"
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            break
    
    if image is not None:
        source_landmarks = sample_row.iloc[:30].values.astype(float).reshape(15, 2)
        source_landmarks[:, 0] *= 299 / 64
        source_landmarks[:, 1] *= 299 / 64
        
        # Use mean normal shape as target
        target_landmarks = pathology_analysis['mean_shapes']['Normal']
        
        # Compare methods
        morphed_images = analyzer.compare_morphing_methods(image, source_landmarks, target_landmarks)
        
        # Visualize comparison
        fig, axes = plt.subplots(1, len(morphed_images), figsize=(15, 5))
        for idx, (method, img) in enumerate(morphed_images.items()):
            if img is not None:
                axes[idx].imshow(img, cmap='gray')
                axes[idx].set_title(f'{method.upper()} Morphing')
                axes[idx].axis('off')
        
        plt.suptitle('Morphing Method Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'morphing_methods_comparison.png', dpi=150)
        plt.close()
    
    print(f"\n✓ Advanced analysis completed!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()