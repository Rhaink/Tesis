#!/usr/bin/env python3
"""
Integration of Delaunay morphing with Template Matching results.

This script leverages the existing Template Matching results (5.63 ± 1.03 pixels error)
to perform morphing analysis using the accurately detected landmarks.
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

# Add project paths
PROJECT_ROOT = Path("/home/donrobot/Projects/Tesis")
sys.path.append(str(PROJECT_ROOT / "delaunay_morphing/src"))
sys.path.append(str(PROJECT_ROOT / "template_matching/src"))

from core.delaunay_lung_morpher import DelaunayLungMorpher, LungShapeMorphingAnalyzer


def load_template_matching_results(results_path: Path) -> Dict:
    """Load the Template Matching results with 5.63px average error."""
    with open(results_path, 'rb') as f:
        return pickle.load(f)


def analyze_morphing_with_tm_results():
    """Analyze morphing using Template Matching detected landmarks."""
    print("=== Morphing Analysis with Template Matching Results ===")
    print("Using landmarks detected with 5.63 ± 1.03 pixels average error\n")
    
    # Load Template Matching results
    results_path = PROJECT_ROOT / "template_matching/results/results_coordenadas_prueba_1.pkl"
    if not results_path.exists():
        print(f"Error: Template Matching results not found at {results_path}")
        return
        
    tm_results = load_template_matching_results(results_path)
    print(f"Loaded {len(tm_results['predictions'])} Template Matching predictions")
    
    # Initialize morpher
    morpher = DelaunayLungMorpher(num_landmarks=15)
    analyzer = LungShapeMorphingAnalyzer(morpher)
    
    # Group results by pathology
    pathology_data = {
        'COVID': {'landmarks': [], 'errors': []},
        'Normal': {'landmarks': [], 'errors': []},
        'Viral Pneumonia': {'landmarks': [], 'errors': []}
    }
    
    # Extract data from the results structure
    predictions = tm_results['predictions']
    ground_truth = tm_results['ground_truth'] 
    errors = tm_results['errors']
    
    # Load CSV to get image names
    csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    df = pd.read_csv(csv_path, header=None)
    
    for idx, (prediction, gt, error) in enumerate(zip(predictions, ground_truth, errors)):
        if idx < len(df):
            image_name = str(df.iloc[idx, -1])  # Last column is image name
            
            # Determine pathology
            for pathology in pathology_data.keys():
                if pathology.replace(' ', '-') in image_name:
                    pathology_data[pathology]['landmarks'].append(prediction)
                    pathology_data[pathology]['errors'].append(error)
                    break
    
    # Compute statistics
    print("\nTemplate Matching Performance by Pathology:")
    for pathology, data in pathology_data.items():
        if data['errors']:
            mean_error = np.mean(data['errors'])
            std_error = np.std(data['errors'])
            print(f"{pathology}: {mean_error:.2f} ± {std_error:.2f} pixels ({len(data['errors'])} images)")
    
    # Select best predictions for morphing demonstration
    print("\n\nSelecting best predictions for morphing demonstration...")
    
    # Find best predictions for each pathology
    best_examples = {}
    for pathology, data in pathology_data.items():
        if data['errors']:
            best_idx = np.argmin(data['errors'])
            best_examples[pathology] = {
                'landmarks': data['landmarks'][best_idx],
                'error': data['errors'][best_idx]
            }
            print(f"{pathology}: Best error = {data['errors'][best_idx]:.2f} pixels")
    
    # Create morphing between best examples
    if len(best_examples) >= 2:
        print("\n\nCreating morphing sequences between best detections...")
        
        # COVID to Normal morphing
        if 'COVID' in best_examples and 'Normal' in best_examples:
            create_morphing_visualization(
                morpher,
                best_examples['COVID']['landmarks'],
                best_examples['Normal']['landmarks'],
                'COVID → Normal (Best Detections)',
                PROJECT_ROOT / 'delaunay_morphing/tm_morphing_covid_normal.png'
            )
        
        # Normal to Viral Pneumonia morphing
        if 'Normal' in best_examples and 'Viral Pneumonia' in best_examples:
            create_morphing_visualization(
                morpher,
                best_examples['Normal']['landmarks'],
                best_examples['Viral Pneumonia']['landmarks'],
                'Normal → Viral Pneumonia (Best Detections)',
                PROJECT_ROOT / 'delaunay_morphing/tm_morphing_normal_viral.png'
            )
    
    # Analyze morphing trajectories with accurate landmarks
    print("\n\nAnalyzing morphing trajectories with Template Matching landmarks...")
    
    # Compute mean shapes from TM results
    mean_shapes_tm = {}
    for pathology, data in pathology_data.items():
        if data['landmarks']:
            mean_shapes_tm[pathology] = np.mean(data['landmarks'], axis=0)
    
    # Compare morphing distances
    print("\nMorphing distances between mean shapes (Template Matching):")
    for p1 in mean_shapes_tm:
        for p2 in mean_shapes_tm:
            if p1 < p2:
                diff = analyzer.compute_shape_difference(mean_shapes_tm[p1], mean_shapes_tm[p2])
                print(f"\n{p1} vs {p2}:")
                print(f"  Mean distance: {diff['mean_distance']:.2f} pixels")
                print(f"  Procrustes distance: {diff['procrustes_mean']:.2f} pixels")
                
                # Analyze trajectory
                trajectory = analyzer.analyze_morphing_trajectory(
                    mean_shapes_tm[p1],
                    mean_shapes_tm[p2],
                    num_steps=10
                )
                total_energy = np.sum(trajectory['deformation_energy'])
                print(f"  Total deformation energy: {total_energy:.2f}")


def create_morphing_visualization(morpher: DelaunayLungMorpher,
                                 landmarks1: np.ndarray,
                                 landmarks2: np.ndarray,
                                 title: str,
                                 output_path: Path):
    """Create visualization of morphing between two landmark sets."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for idx, (ax, alpha) in enumerate(zip(axes, alphas)):
        # Interpolate landmarks
        current_landmarks = (1 - alpha) * landmarks1 + alpha * landmarks2
        
        # Create blank canvas
        canvas = np.ones((299, 299)) * 255
        
        # Draw anatomical connections
        # Contour connections
        contour_connections = [
            (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
            (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
        ]
        
        # Draw connections on canvas
        for i, j in contour_connections:
            pt1 = tuple(current_landmarks[i].astype(int))
            pt2 = tuple(current_landmarks[j].astype(int))
            cv2.line(canvas, pt1, pt2, 0, 2)
        
        # Mediastinal connections
        mediastinal_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
        for i, j in mediastinal_connections:
            pt1 = tuple(current_landmarks[i].astype(int))
            pt2 = tuple(current_landmarks[j].astype(int))
            cv2.line(canvas, pt1, pt2, 128, 2)
        
        # Draw landmarks
        for landmark in current_landmarks:
            cv2.circle(canvas, tuple(landmark.astype(int)), 4, 0, -1)
        
        ax.imshow(canvas, cmap='gray')
        ax.set_title(f'α = {alpha}')
        ax.axis('off')
        
        # Add triangulation overlay for middle frame
        if alpha == 0.5:
            tri = morpher.create_triangulation(current_landmarks, add_boundary=False)
            for simplex in tri.simplices:
                triangle = current_landmarks[simplex]
                triangle = np.vstack([triangle, triangle[0]])
                ax.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.5, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def analyze_deformation_consistency():
    """Analyze consistency of deformations across the dataset."""
    print("\n\n=== Deformation Consistency Analysis ===")
    
    # Load Template Matching results
    results_path = PROJECT_ROOT / "template_matching/results/results_coordenadas_prueba_1.pkl"
    tm_results = load_template_matching_results(results_path)
    
    # Load ground truth
    csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    df = pd.read_csv(csv_path, header=None)
    
    morpher = DelaunayLungMorpher()
    analyzer = LungShapeMorphingAnalyzer(morpher)
    
    # Analyze deformation fields for each pathology transition
    deformation_stats = {}
    
    # Get mean shape from all data as reference
    all_landmarks = []
    for _, row in df.iterrows():
        coords = row.iloc[:30].values.astype(float).reshape(15, 2)
        coords[:, 0] *= 299 / 64
        coords[:, 1] *= 299 / 64
        all_landmarks.append(coords)
    
    reference_shape = np.mean(all_landmarks, axis=0)
    
    # Compute deformation fields to reference shape
    print("\nComputing deformation fields to reference shape...")
    
    pathology_deformations = {
        'COVID': [],
        'Normal': [],
        'Viral Pneumonia': []
    }
    
    for idx, prediction in enumerate(predictions):
        if idx < len(df):
            image_name = str(df.iloc[idx, -1])
            predicted_landmarks = prediction
        
        # Compute deformation field
        deform_field = analyzer.analyze_deformation_field(
            predicted_landmarks,
            reference_shape,
            (299, 299)
        )
        
        # Compute magnitude statistics
        magnitude = np.sqrt(deform_field[:, :, 0]**2 + deform_field[:, :, 1]**2)
        
        # Store by pathology
        for pathology in pathology_deformations.keys():
            if pathology.replace(' ', '-') in image_name:
                pathology_deformations[pathology].append({
                    'mean_magnitude': np.mean(magnitude),
                    'max_magnitude': np.max(magnitude),
                    'std_magnitude': np.std(magnitude)
                })
                break
    
    # Compute statistics
    print("\nDeformation Statistics by Pathology:")
    for pathology, deforms in pathology_deformations.items():
        if deforms:
            mean_mags = [d['mean_magnitude'] for d in deforms]
            print(f"\n{pathology}:")
            print(f"  Mean deformation: {np.mean(mean_mags):.2f} ± {np.std(mean_mags):.2f} pixels")
            print(f"  Range: [{np.min(mean_mags):.2f}, {np.max(mean_mags):.2f}]")


def main():
    """Run Template Matching integration analysis."""
    print("Delaunay Morphing Integration with Template Matching Results")
    print("==========================================================\n")
    
    # Create output directory
    output_dir = PROJECT_ROOT / 'delaunay_morphing'
    output_dir.mkdir(exist_ok=True)
    
    # Run analyses
    analyze_morphing_with_tm_results()
    analyze_deformation_consistency()
    
    print("\n✓ Template Matching integration completed!")


if __name__ == "__main__":
    main()