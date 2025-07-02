#!/usr/bin/env python3
"""
Create comparative visualization showing Ground Truth vs Template Matching vs Geometric predictions.
"""

import os
import sys
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = '/home/donrobot/Projects/Tesis'
sys.path.append(os.path.join(PROJECT_ROOT, 'matching_geometric/src/core'))
sys.path.append(os.path.join(PROJECT_ROOT, 'pulmones/src'))

from geometric_predictor import GeometricLandmarkPredictor
from utils import asm_utils


def load_data():
    """Load Template Matching results and initialize geometric predictor."""
    print("Loading data...")
    
    # Load TM results
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    with open(results_file, 'rb') as f:
        tm_results = pickle.load(f)
    
    # Initialize geometric predictor
    model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    geo_predictor = GeometricLandmarkPredictor(model_path)
    
    return tm_results, geo_predictor


def draw_landmarks_on_image(image, landmarks, color, size=4, draw_lines=True):
    """Draw landmarks on image with optional connecting lines."""
    img_vis = image.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    
    # Draw landmarks
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_vis, (int(x), int(y)), size, color, -1)
        # Add landmark numbers
        cv2.putText(img_vis, str(i), (int(x)+5, int(y)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw anatomical connections
    if draw_lines:
        connections = [(0,12), (12,3), (3,5), (5,7), (7,14), (14,1), 
                      (1,13), (13,6), (6,4), (4,2), (2,11), (11,0),
                      (0,8), (8,9), (9,10), (10,1)]
        
        for p1, p2 in connections:
            if p1 < len(landmarks) and p2 < len(landmarks):
                pt1 = (int(landmarks[p1][0]), int(landmarks[p1][1]))
                pt2 = (int(landmarks[p2][0]), int(landmarks[p2][1]))
                cv2.line(img_vis, pt1, pt2, color, 1)
    
    return img_vis


def create_single_comparison(image, image_name, gt_landmarks, tm_landmarks, geo_predictor):
    """Create comparison for a single image."""
    # Get geometric prediction
    geo_result = geo_predictor.predict_landmarks(image, image_name=image_name)
    
    # For geometric, we only show the 5 key points (main 2 + 3 quartiles)
    geo_landmarks_full = geo_result['landmarks']
    geo_quartiles = geo_result['intermediate_points']
    
    # Create 5-point geometric visualization
    geo_5_points = np.array([
        geo_landmarks_full[0],  # Point 0
        geo_landmarks_full[1],  # Point 1
        geo_quartiles['cuarto1'],  # Quartile 1
        geo_quartiles['medio'],    # Middle
        geo_quartiles['cuarto3']   # Quartile 3
    ])
    
    # Create visualizations
    gt_vis = draw_landmarks_on_image(image, gt_landmarks, (0, 255, 0), size=4)      # Green
    tm_vis = draw_landmarks_on_image(image, tm_landmarks, (255, 0, 0), size=4)      # Red  
    geo_vis = draw_landmarks_on_image(image, geo_5_points, (0, 255, 255), size=5, draw_lines=False)  # Yellow, no lines
    
    # Add main line for geometric
    cv2.line(geo_vis, 
             tuple(geo_landmarks_full[0].astype(int)), 
             tuple(geo_landmarks_full[1].astype(int)), 
             (0, 255, 0), 2)  # Green line
    
    return gt_vis, tm_vis, geo_vis


def create_comparison_grid(images_to_process=6):
    """Create a grid comparison of multiple images."""
    print(f"Creating comparison grid for {images_to_process} images...")
    
    # Load data
    tm_results, geo_predictor = load_data()
    
    image_names = tm_results['image_names'][:images_to_process]
    tm_predictions = tm_results['predictions'][:images_to_process]
    ground_truth = tm_results['ground_truth'][:images_to_process]
    
    # Load images and create comparisons
    comparisons = []
    images_base_dir = os.path.join(PROJECT_ROOT, 'COVID-19_Radiography_Dataset')
    
    for i, (image_name, tm_pred, gt) in enumerate(zip(image_names, tm_predictions, ground_truth)):
        try:
            print(f"Processing {i+1}/{images_to_process}: {image_name}")
            
            # Load image
            img_path = asm_utils.get_image_path(image_name, None, images_base_dir)
            image = asm_utils.load_image_grayscale(img_path)
            
            # Create comparison
            gt_vis, tm_vis, geo_vis = create_single_comparison(image, image_name, gt, tm_pred, geo_predictor)
            
            comparisons.append({
                'name': image_name,
                'gt': gt_vis,
                'tm': tm_vis,
                'geo': geo_vis
            })
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue
    
    # Create grid visualization
    fig, axes = plt.subplots(len(comparisons), 3, figsize=(15, 5*len(comparisons)))
    
    if len(comparisons) == 1:
        axes = axes.reshape(1, -1)
    
    for i, comp in enumerate(comparisons):
        # Ground Truth
        axes[i, 0].imshow(comp['gt'][:, :, ::-1])  # Convert BGR to RGB
        axes[i, 0].set_title(f'Ground Truth\n{comp["name"]}', fontweight='bold')
        axes[i, 0].axis('off')
        
        # Template Matching
        axes[i, 1].imshow(comp['tm'][:, :, ::-1])
        axes[i, 1].set_title('Template Matching\n(All 15 landmarks)', fontweight='bold')
        axes[i, 1].axis('off')
        
        # Geometric
        axes[i, 2].imshow(comp['geo'][:, :, ::-1])
        axes[i, 2].set_title('Matching Geometric\n(5 key points)', fontweight='bold')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'method_comparison_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison grid saved: {output_path}")
    
    plt.close()
    
    return output_path


def create_detailed_single_comparison():
    """Create detailed comparison for a single representative image."""
    print("Creating detailed single image comparison...")
    
    # Load data
    tm_results, geo_predictor = load_data()
    
    # Use Normal-3173 as example
    target_name = 'Normal-3173'
    found_idx = None
    
    for i, name in enumerate(tm_results['image_names']):
        if target_name in name:
            found_idx = i
            break
    
    if found_idx is None:
        print(f"Image {target_name} not found, using first image")
        found_idx = 0
    
    image_name = tm_results['image_names'][found_idx]
    tm_pred = tm_results['predictions'][found_idx]
    gt = tm_results['ground_truth'][found_idx]
    
    # Load image
    images_base_dir = os.path.join(PROJECT_ROOT, 'COVID-19_Radiography_Dataset')
    img_path = asm_utils.get_image_path(image_name, None, images_base_dir)
    image = asm_utils.load_image_grayscale(img_path)
    
    # Create detailed comparison
    gt_vis, tm_vis, geo_vis = create_single_comparison(image, image_name, gt, tm_pred, geo_predictor)
    
    # Calculate errors for this image
    tm_errors = np.linalg.norm(tm_pred - gt, axis=1)
    geo_result = geo_predictor.predict_landmarks(image, image_name=image_name)
    
    # Calculate errors for quartile points
    quartile_mapping = {8: 'cuarto1', 9: 'medio', 10: 'cuarto3'}
    quartile_errors = {}
    
    for landmark_id, quartile_name in quartile_mapping.items():
        tm_point = tm_pred[landmark_id]
        gt_point = gt[landmark_id]
        geo_point = geo_result['intermediate_points'][quartile_name]
        
        tm_error = np.linalg.norm(tm_point - gt_point)
        geo_error = np.linalg.norm(geo_point - gt_point)
        
        quartile_errors[quartile_name] = {
            'tm_error': tm_error,
            'geo_error': geo_error,
            'improvement': ((tm_error - geo_error) / tm_error) * 100
        }
    
    # Create detailed figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: visualizations
    axes[0, 0].imshow(gt_vis[:, :, ::-1])
    axes[0, 0].set_title(f'Ground Truth\n{image_name}', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tm_vis[:, :, ::-1])
    axes[0, 1].set_title('Template Matching\nAll 15 landmarks', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(geo_vis[:, :, ::-1])
    axes[0, 2].set_title('Matching Geometric\n5 key points', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Bottom row: error analysis
    # Error comparison for quartile points
    quartile_names = list(quartile_errors.keys())
    tm_quartile_errors = [quartile_errors[q]['tm_error'] for q in quartile_names]
    geo_quartile_errors = [quartile_errors[q]['geo_error'] for q in quartile_names]
    
    x = np.arange(len(quartile_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, tm_quartile_errors, width, label='Template Matching', alpha=0.8, color='red')
    axes[1, 0].bar(x + width/2, geo_quartile_errors, width, label='Geometric', alpha=0.8, color='gold')
    axes[1, 0].set_xlabel('Quartile Points')
    axes[1, 0].set_ylabel('Error (pixels)')
    axes[1, 0].set_title('Quartile Points Error Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(quartile_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Improvement percentages
    improvements = [quartile_errors[q]['improvement'] for q in quartile_names]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    axes[1, 1].bar(quartile_names, improvements, alpha=0.8, color=colors)
    axes[1, 1].set_xlabel('Quartile Points')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title('Geometric Method Improvement')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Error distribution for all landmarks
    axes[1, 2].bar(range(15), tm_errors, alpha=0.8, color='red', label='Template Matching')
    axes[1, 2].set_xlabel('Landmark Index')
    axes[1, 2].set_ylabel('Error (pixels)')
    axes[1, 2].set_title('All Landmarks Error (TM)')
    axes[1, 2].set_xticks(range(15))
    axes[1, 2].grid(True, alpha=0.3)
    
    # Highlight quartile points
    for landmark_id in [8, 9, 10]:
        axes[1, 2].bar(landmark_id, tm_errors[landmark_id], alpha=1.0, color='darkred')
    
    plt.tight_layout()
    
    # Save detailed comparison
    output_path = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations/detailed_method_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Detailed comparison saved: {output_path}")
    
    plt.close()
    
    # Print summary
    print(f"\n📊 DETAILED ANALYSIS FOR {image_name}:")
    print(f"Overall TM error: {tm_errors.mean():.3f} ± {tm_errors.std():.3f} pixels")
    print(f"\nQuartile points comparison:")
    for q_name, errors in quartile_errors.items():
        print(f"  {q_name}: TM {errors['tm_error']:.3f}px → Geo {errors['geo_error']:.3f}px ({errors['improvement']:+.1f}%)")
    
    return output_path


def main():
    """Create all comparison visualizations."""
    print("="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    # Create grid comparison
    grid_path = create_comparison_grid(images_to_process=6)
    
    # Create detailed single comparison
    detailed_path = create_detailed_single_comparison()
    
    print(f"\n✅ Comparison visualizations completed!")
    print(f"📊 Grid comparison: {grid_path}")
    print(f"📊 Detailed comparison: {detailed_path}")


if __name__ == "__main__":
    main()