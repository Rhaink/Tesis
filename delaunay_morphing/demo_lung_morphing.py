#!/usr/bin/env python3
"""
Demonstration of Delaunay-based lung morphing using the COVID-19 dataset.

This script shows how to use the DelaunayLungMorpher to perform morphing
between different lung shapes from the dataset.
"""

import sys
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional

# Add project paths
PROJECT_ROOT = Path("/home/donrobot/Projects/Tesis")
sys.path.append(str(PROJECT_ROOT / "delaunay_morphing/src"))
sys.path.append(str(PROJECT_ROOT / "pulmones/src"))

from core.delaunay_lung_morpher import DelaunayLungMorpher, LungShapeMorphingAnalyzer


def load_landmarks_from_csv(csv_path: Path, image_name: str) -> Optional[np.ndarray]:
    """Load landmarks for a specific image from CSV file."""
    # Read CSV without header
    df = pd.read_csv(csv_path, header=None)
    
    # Remove .png extension if present for comparison
    image_name_no_ext = image_name.replace('.png', '')
    
    # Find the row with the image name
    for idx, row in df.iterrows():
        csv_image_name = str(row.iloc[-1]).replace('.png', '')
        if csv_image_name == image_name_no_ext:
            # Extract coordinates (first 30 values)
            coords = row.iloc[:30].values.astype(float)
            landmarks = coords.reshape(15, 2)
            return landmarks
    
    return None


def scale_landmarks(landmarks: np.ndarray, 
                   from_size: Tuple[int, int] = (64, 64),
                   to_size: Tuple[int, int] = (299, 299)) -> np.ndarray:
    """Scale landmarks from reference coordinate system to image size."""
    scaled = landmarks.copy()
    scaled[:, 0] *= to_size[0] / from_size[0]
    scaled[:, 1] *= to_size[1] / from_size[1]
    return scaled


def load_lung_image(image_name: str) -> Optional[np.ndarray]:
    """Load lung X-ray image from dataset."""
    # Define possible paths
    categories = ['COVID', 'Normal', 'Viral Pneumonia']
    
    for category in categories:
        image_path = PROJECT_ROOT / f"COVID-19_Radiography_Dataset/{category}/images/{image_name}"
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            return image
    
    return None


def demonstrate_basic_morphing():
    """Demonstrate basic morphing between two lung shapes."""
    print("=== Basic Morphing Demonstration ===")
    
    # Initialize morpher
    morpher = DelaunayLungMorpher(num_landmarks=15)
    analyzer = LungShapeMorphingAnalyzer(morpher)
    
    # Load test dataset
    csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    df = pd.read_csv(csv_path, header=None)
    
    # Select two different lung shapes
    image1_name = "COVID-3519.png"  # First COVID image in test set
    image2_name = "Normal-1756.png"  # Best performing image
    
    # Load landmarks
    landmarks1 = load_landmarks_from_csv(csv_path, image1_name)
    landmarks2 = load_landmarks_from_csv(csv_path, image2_name)
    
    if landmarks1 is None or landmarks2 is None:
        print("Error: Could not load landmarks")
        return
    
    # Load images
    image1 = load_lung_image(image1_name)
    image2 = load_lung_image(image2_name)
    
    if image1 is None or image2 is None:
        print("Error: Could not load images")
        return
    
    # Scale landmarks to image size
    landmarks1_scaled = scale_landmarks(landmarks1)
    landmarks2_scaled = scale_landmarks(landmarks2)
    
    # Perform morphing at different stages
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    morphed_images = []
    
    for alpha in alphas:
        result = morpher.morph_image(
            image1,
            landmarks1_scaled,
            landmarks2_scaled,
            alpha=alpha
        )
        morphed_images.append(result.warped_image)
    
    # Visualize results
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ['Source (COVID)', 'Alpha=0.25', 'Alpha=0.5', 'Alpha=0.75', 'Target (Normal)']
    
    for idx, (ax, img, title) in enumerate(zip(axes, morphed_images, titles)):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle('Lung Shape Morphing Sequence')
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'delaunay_morphing/morphing_sequence.png', dpi=150)
    plt.show()
    
    # Compute shape difference metrics
    shape_diff = analyzer.compute_shape_difference(landmarks1_scaled, landmarks2_scaled)
    print("\nShape Difference Metrics:")
    for key, value in shape_diff.items():
        print(f"  {key}: {value:.2f}")


def demonstrate_triangulation_visualization():
    """Visualize Delaunay triangulation on lung landmarks."""
    print("\n=== Triangulation Visualization ===")
    
    morpher = DelaunayLungMorpher()
    
    # Load a sample image and landmarks
    csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    image_name = "Normal-1756.png"  # Best performing image
    
    landmarks = load_landmarks_from_csv(csv_path, image_name)
    image = load_lung_image(image_name)
    
    if landmarks is None or image is None:
        print("Error: Could not load data")
        return
    
    # Scale landmarks
    landmarks_scaled = scale_landmarks(landmarks)
    
    # Create and visualize triangulation
    fig = morpher.visualize_triangulation(landmarks_scaled, image, 
                                        title=f"Delaunay Triangulation - {image_name}")
    plt.savefig(PROJECT_ROOT / 'delaunay_morphing/triangulation_viz.png', dpi=150)
    plt.show()
    
    # Compute triangulation quality metrics
    tri = morpher.create_triangulation(landmarks_scaled, add_boundary=False)
    metrics = morpher.compute_triangle_quality_metrics(tri)
    
    print("\nTriangulation Quality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")


def demonstrate_deformation_field():
    """Visualize deformation field between lung shapes."""
    print("\n=== Deformation Field Analysis ===")
    
    morpher = DelaunayLungMorpher()
    analyzer = LungShapeMorphingAnalyzer(morpher)
    
    # Load two different lung shapes
    csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    image1_name = "COVID-3519.png"  # First COVID image in test set
    image2_name = "Normal-1756.png"  # Best performing image
    
    landmarks1 = load_landmarks_from_csv(csv_path, image1_name)
    landmarks2 = load_landmarks_from_csv(csv_path, image2_name)
    
    if landmarks1 is None or landmarks2 is None:
        print("Error: Could not load landmarks")
        return
    
    # Scale landmarks
    landmarks1_scaled = scale_landmarks(landmarks1)
    landmarks2_scaled = scale_landmarks(landmarks2)
    
    # Compute deformation field
    deform_field = analyzer.analyze_deformation_field(
        landmarks1_scaled, 
        landmarks2_scaled,
        (299, 299)
    )
    
    # Visualize deformation field
    fig = analyzer.visualize_deformation_field(deform_field, downsample=15)
    plt.savefig(PROJECT_ROOT / 'delaunay_morphing/deformation_field.png', dpi=150)
    plt.show()
    
    # Compute statistics
    magnitude = np.sqrt(deform_field[:, :, 0]**2 + deform_field[:, :, 1]**2)
    print(f"\nDeformation Field Statistics:")
    print(f"  Mean displacement: {np.mean(magnitude):.2f} pixels")
    print(f"  Max displacement: {np.max(magnitude):.2f} pixels")
    print(f"  Std displacement: {np.std(magnitude):.2f} pixels")


def demonstrate_morphing_animation():
    """Create an animation of morphing between lung shapes."""
    print("\n=== Creating Morphing Animation ===")
    
    morpher = DelaunayLungMorpher()
    
    # Load images and landmarks
    csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    image1_name = "COVID-259.png"
    image2_name = "Normal-1756.png"
    
    landmarks1 = load_landmarks_from_csv(csv_path, image1_name)
    landmarks2 = load_landmarks_from_csv(csv_path, image2_name)
    image1 = load_lung_image(image1_name)
    
    if any(x is None for x in [landmarks1, landmarks2, image1]):
        print("Error: Could not load data")
        return
    
    # Scale landmarks
    landmarks1_scaled = scale_landmarks(landmarks1)
    landmarks2_scaled = scale_landmarks(landmarks2)
    
    # Create morphing sequence
    frames = morpher.create_morphing_sequence(
        image1,
        landmarks1_scaled,
        landmarks2_scaled,
        num_frames=20
    )
    
    # Save as video
    output_path = PROJECT_ROOT / 'delaunay_morphing/morphing_animation.mp4'
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height), False)
    
    for frame in frames:
        # Convert to uint8
        frame_uint8 = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
        out.write(frame_uint8)
    
    # Add reverse sequence for loop
    for frame in reversed(frames[1:-1]):
        frame_uint8 = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
        out.write(frame_uint8)
    
    out.release()
    print(f"Animation saved to: {output_path}")


def compare_pathology_shapes():
    """Compare average lung shapes across different pathologies."""
    print("\n=== Pathology Shape Comparison ===")
    
    morpher = DelaunayLungMorpher()
    analyzer = LungShapeMorphingAnalyzer(morpher)
    
    # Load dataset
    csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    df = pd.read_csv(csv_path, header=None)
    
    # Group by pathology
    pathologies = {
        'COVID': [],
        'Normal': [],
        'Viral Pneumonia': []
    }
    
    for _, row in df.iterrows():
        image_name = row.iloc[-1]
        coords = row.iloc[:30].values.astype(float).reshape(15, 2)
        scaled_coords = scale_landmarks(coords)
        
        for pathology in pathologies.keys():
            if pathology.replace(' ', '-') in image_name:
                pathologies[pathology].append(scaled_coords)
                break
    
    # Compute mean shapes
    mean_shapes = {}
    for pathology, shapes in pathologies.items():
        if shapes:
            mean_shapes[pathology] = np.mean(shapes, axis=0)
    
    # Visualize mean shapes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colors = {'COVID': 'red', 'Normal': 'green', 'Viral Pneumonia': 'blue'}
    
    for pathology, mean_shape in mean_shapes.items():
        # Plot landmarks
        ax.scatter(mean_shape[:, 0], mean_shape[:, 1], 
                  c=colors[pathology], s=100, label=pathology, alpha=0.7)
        
        # Plot anatomically correct connections
        # Contour connections
        contour_connections = [(0,12), (12,3), (3,5), (5,7), (7,14), (14,1), 
                              (1,13), (13,6), (6,4), (4,2), (2,11), (11,0)]
        
        for i, j in contour_connections:
            ax.plot([mean_shape[i, 0], mean_shape[j, 0]], 
                   [mean_shape[i, 1], mean_shape[j, 1]], 
                   c=colors[pathology], linewidth=2, alpha=0.7)
                   
        # Mediastinal connections
        mediastinal_connections = [(0,8), (8,9), (9,10), (10,1)]
        for i, j in mediastinal_connections:
            ax.plot([mean_shape[i, 0], mean_shape[j, 0]], 
                   [mean_shape[i, 1], mean_shape[j, 1]], 
                   c=colors[pathology], linewidth=2, alpha=0.7, linestyle='--')
    
    ax.set_title('Mean Lung Shapes by Pathology')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'delaunay_morphing/pathology_comparison.png', dpi=150)
    plt.show()
    
    # Compute pairwise shape differences
    print("\nPairwise Shape Differences:")
    for p1 in mean_shapes:
        for p2 in mean_shapes:
            if p1 < p2:  # Avoid duplicates
                diff = analyzer.compute_shape_difference(mean_shapes[p1], mean_shapes[p2])
                print(f"\n{p1} vs {p2}:")
                print(f"  Mean distance: {diff['mean_distance']:.2f} pixels")
                print(f"  Procrustes distance: {diff['procrustes_mean']:.2f} pixels")


def main():
    """Run all demonstrations."""
    # Create output directory
    output_dir = PROJECT_ROOT / 'delaunay_morphing'
    output_dir.mkdir(exist_ok=True)
    
    print("Delaunay Lung Morphing Demonstration")
    print("====================================\n")
    
    # Run demonstrations
    demonstrate_basic_morphing()
    demonstrate_triangulation_visualization()
    demonstrate_deformation_field()
    demonstrate_morphing_animation()
    compare_pathology_shapes()
    
    print("\n✓ All demonstrations completed!")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()