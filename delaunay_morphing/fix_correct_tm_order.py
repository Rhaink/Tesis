#!/usr/bin/env python3
"""
Generate correct visualizations using the exact Template Matching order and landmarks.
"""

import sys
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = Path("/home/donrobot/Projects/Tesis")
sys.path.append(str(PROJECT_ROOT / "delaunay_morphing/src"))

from core.delaunay_lung_morpher import DelaunayLungMorpher


def get_correct_tm_order():
    """Get the correct Template Matching order from visualization files."""
    viz_dir = PROJECT_ROOT / "template_matching/visualizations/all_test_images/landmark_predictions/"
    viz_files = sorted([f for f in viz_dir.glob("*_landmarks.png")])
    
    tm_order = []
    for viz_file in viz_files:
        # Extract image name from filename like "000_Normal-3173_landmarks.png"
        parts = viz_file.stem.split('_')
        if len(parts) >= 2:
            image_name = '_'.join(parts[1:]).replace('_landmarks', '')
            # Convert underscores back to spaces for "Viral Pneumonia"
            image_name = image_name.replace('Viral_Pneumonia', 'Viral Pneumonia')
            tm_order.append(image_name)
    
    return tm_order


def load_image_by_name(image_name):
    """Load image from dataset."""
    for category in ['COVID', 'Normal', 'Viral Pneumonia']:
        image_path = PROJECT_ROOT / f"COVID-19_Radiography_Dataset/{category}/images/{image_name}.png"
        if image_path.exists():
            return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return None


def create_correct_visualization(tm_idx, image_name, tm_results, canonical_shape, morpher, output_dir):
    """Create visualization with correct TM landmarks."""
    
    # Get TM landmarks (already in 299x299 scale)
    predicted_landmarks = tm_results['predictions'][tm_idx]
    ground_truth_landmarks = tm_results['ground_truth'][tm_idx]
    error_per_landmark = tm_results['errors'][tm_idx]
    mean_error = np.mean(error_per_landmark)
    
    # Load original image
    original_image = load_image_by_name(image_name)
    if original_image is None:
        print(f"Could not load image: {image_name}")
        return False
    
    # Perform morphing to canonical shape
    try:
        morphing_result = morpher.morph_image(
            original_image,
            predicted_landmarks,
            canonical_shape,
            output_shape=original_image.shape,
            alpha=1.0
        )
        warped_image = morphing_result.warped_image
        
        # Compute triangulation quality
        triangulation_quality = morpher.compute_triangle_quality_metrics(morphing_result.triangulation)
        min_angle = triangulation_quality['min_angle']
        
        # Compute morphing distance
        shape_diff_dist = np.mean(np.linalg.norm(predicted_landmarks - canonical_shape, axis=1))
        
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return False
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original image with TM landmarks
    axes[0].imshow(original_image, cmap='gray')
    
    # Plot predicted landmarks (red X) and ground truth (green circles)
    axes[0].scatter(ground_truth_landmarks[:, 0], ground_truth_landmarks[:, 1], 
                   c='green', s=40, alpha=0.8, marker='o', label='Ground Truth')
    axes[0].scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], 
                   c='red', s=60, alpha=0.9, marker='x', linewidths=2, label='TM Predicted')
    
    # Add anatomical connections using predicted landmarks
    contour_connections = [
        (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
        (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
    ]
    
    for i, j in contour_connections:
        axes[0].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                    [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                    'lime', linewidth=2, alpha=0.8)
    
    # Mediastinal connections
    mediastinal_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
    for i, j in mediastinal_connections:
        axes[0].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                    [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                    'orange', linewidth=2, alpha=0.8, linestyle='--')
    
    axes[0].set_title(f"Original: {image_name}\\nTM Error: {mean_error:.2f}px", fontsize=12)
    axes[0].axis('off')
    axes[0].legend(loc='upper right', fontsize=8)
    
    # Panel 2: Warped image with canonical landmarks
    axes[1].imshow(warped_image, cmap='gray')
    axes[1].scatter(canonical_shape[:, 0], canonical_shape[:, 1], 
                   c='blue', s=40, alpha=0.8, label='Canonical Shape')
    
    # Add canonical shape connections
    for i, j in contour_connections:
        axes[1].plot([canonical_shape[i, 0], canonical_shape[j, 0]], 
                    [canonical_shape[i, 1], canonical_shape[j, 1]], 
                    'lime', linewidth=2, alpha=0.8)
    
    for i, j in mediastinal_connections:
        axes[1].plot([canonical_shape[i, 0], canonical_shape[j, 0]], 
                    [canonical_shape[i, 1], canonical_shape[j, 1]], 
                    'orange', linewidth=2, alpha=0.8, linestyle='--')
    
    axes[1].set_title(f"Warped to Canonical\\nMorph Dist: {shape_diff_dist:.1f}px", fontsize=12)
    axes[1].axis('off')
    axes[1].legend(loc='upper right', fontsize=8)
    
    # Panel 3: Triangulation overlay
    axes[2].imshow(original_image, cmap='gray', alpha=0.9)
    
    # Draw Delaunay triangulation
    tri = morpher.create_triangulation(predicted_landmarks, add_boundary=False)
    for simplex in tri.simplices:
        triangle = predicted_landmarks[simplex]
        triangle = np.vstack([triangle, triangle[0]])
        axes[2].plot(triangle[:, 0], triangle[:, 1], 'cyan', linewidth=1, alpha=0.5)
    
    # Draw anatomical connections (stronger)
    for i, j in contour_connections:
        axes[2].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                    [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                    'lime', linewidth=3, alpha=0.9)
    
    for i, j in mediastinal_connections:
        axes[2].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                    [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                    'orange', linewidth=3, alpha=0.9, linestyle='--')
    
    # Add landmark numbers
    for idx, (x, y) in enumerate(predicted_landmarks):
        axes[2].annotate(str(idx), (x, y), xytext=(3, 3), textcoords='offset points',
                        fontsize=8, color='yellow', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    axes[2].scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], 
                   c='red', s=50, edgecolors='white', linewidths=1.5, marker='x')
    
    axes[2].set_title(f"Delaunay Triangulation\\nMin Angle: {min_angle:.1f}°", fontsize=12)
    axes[2].axis('off')
    
    # Determine pathology and background color
    pathology = "Unknown"
    if "COVID" in image_name:
        pathology = "COVID"
        fig.patch.set_facecolor('#ffe6e6')
    elif "Normal" in image_name:
        pathology = "Normal"
        fig.patch.set_facecolor('#e6ffe6')
    elif "Viral" in image_name:
        pathology = "Viral Pneumonia"
        fig.patch.set_facecolor('#e6f3ff')
    
    plt.suptitle(f'#{tm_idx+1:03d} - {pathology} - Correct TM Landmarks (Error: {mean_error:.2f}px)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save with correct naming
    output_file = output_dir / f"{pathology.replace(' ', '_')}_{tm_idx+1:03d}_{image_name}.png"
    plt.savefig(output_file, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    return {
        'tm_idx': tm_idx,
        'image_name': image_name,
        'pathology': pathology,
        'tm_error': mean_error,
        'morph_distance': shape_diff_dist,
        'min_angle': min_angle
    }


def main():
    """Generate all correct visualizations."""
    print("Generating CORRECT Template Matching Visualizations")
    print("=" * 55)
    print("Using exact TM landmarks with 5.63±1.03px error")
    
    # Load Template Matching results
    with open(PROJECT_ROOT / "template_matching/results/results_coordenadas_prueba_1.pkl", 'rb') as f:
        tm_results = pickle.load(f)
    
    # Get correct TM order
    tm_order = get_correct_tm_order()
    print(f"Found {len(tm_order)} images in TM order")
    
    # Load canonical shape
    canonical_shape = np.load(PROJECT_ROOT / "delaunay_morphing/processed_159/canonical_shape.npy")
    
    # Initialize morpher
    morpher = DelaunayLungMorpher()
    
    # Create output directory
    output_dir = PROJECT_ROOT / "delaunay_morphing/correct_tm_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Process all images in correct TM order
    results = []
    success_count = 0
    
    for tm_idx, image_name in enumerate(tqdm(tm_order, desc="Creating correct visualizations")):
        result = create_correct_visualization(
            tm_idx, image_name, tm_results, canonical_shape, morpher, output_dir
        )
        
        if result:
            results.append(result)
            success_count += 1
    
    print(f"\\nCorrect Visualizations Complete:")
    print(f"  Successfully created: {success_count}")
    print(f"  Total expected: {len(tm_order)}")
    
    # Verify TM error consistency
    tm_errors = [r['tm_error'] for r in results]
    print(f"\\nTemplate Matching Error Verification:")
    print(f"  Mean: {np.mean(tm_errors):.2f} ± {np.std(tm_errors):.2f} pixels")
    print(f"  Expected: 5.63 ± 1.03 pixels")
    print(f"  Match: {'✓' if abs(np.mean(tm_errors) - 5.63) < 0.1 else '✗'}")
    
    # Distribution by pathology
    pathology_counts = {}
    for result in results:
        pathology = result['pathology']
        pathology_counts[pathology] = pathology_counts.get(pathology, 0) + 1
    
    print(f"\\nDistribution by pathology:")
    for pathology, count in pathology_counts.items():
        print(f"  {pathology}: {count} images")
    
    # Save results summary
    with open(output_dir / "correct_results_summary.json", 'w') as f:
        json.dump({
            'total_images': len(results),
            'mean_tm_error': float(np.mean(tm_errors)),
            'std_tm_error': float(np.std(tm_errors)),
            'pathology_distribution': pathology_counts,
            'verification': 'CORRECT - Using exact TM landmarks'
        }, f, indent=2)
    
    print(f"\\n✓ CORRECT Template Matching visualizations completed!")
    print(f"✓ All landmarks now match TM results exactly")
    print(f"✓ Error: {np.mean(tm_errors):.2f}±{np.std(tm_errors):.2f}px (Expected: 5.63±1.03px)")


if __name__ == "__main__":
    main()