#!/usr/bin/env python3
"""
Generate visualizations for all 159 processed images.
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


def load_image_by_name(image_name):
    """Load image from dataset."""
    for category in ['COVID', 'Normal', 'Viral Pneumonia']:
        image_path = PROJECT_ROOT / f"COVID-19_Radiography_Dataset/{category}/images/{image_name}.png"
        if image_path.exists():
            return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return None


def create_visualization_for_image(idx, result_data, morpher, canonical_shape, output_dir):
    """Create visualization for a single image."""
    image_name = result_data['image_name']
    tm_error = result_data['template_matching_error']
    morph_distance = result_data['morphing_distance']
    
    # Load Template Matching results to get landmarks
    results_path = PROJECT_ROOT / "template_matching/results/results_coordenadas_prueba_1.pkl"
    with open(results_path, 'rb') as f:
        tm_results = pickle.load(f)
    
    predicted_landmarks = tm_results['predictions'][idx]
    
    # Load original image
    original_image = load_image_by_name(image_name)
    if original_image is None:
        print(f"Could not load image: {image_name}")
        return False
    
    # Perform morphing to get warped image
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
        
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return False
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Original image with TM landmarks
    axes[0].imshow(original_image, cmap='gray')
    axes[0].scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], 
                   c='red', s=20, alpha=0.8)
    
    # Add anatomical connections
    contour_connections = [
        (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
        (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
    ]
    
    for i, j in contour_connections:
        axes[0].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                    [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                    'g-', linewidth=1.5, alpha=0.7)
    
    # Mediastinal connections
    mediastinal_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
    for i, j in mediastinal_connections:
        axes[0].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                    [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                    'orange', linewidth=1.5, alpha=0.7, linestyle='--')
    
    axes[0].set_title(f"Original: {image_name}\nTM Error: {tm_error:.2f}px")
    axes[0].axis('off')
    
    # Panel 2: Warped image with canonical landmarks
    axes[1].imshow(warped_image, cmap='gray')
    axes[1].scatter(canonical_shape[:, 0], canonical_shape[:, 1], 
                   c='blue', s=20, alpha=0.8)
    
    # Add canonical shape connections
    for i, j in contour_connections:
        axes[1].plot([canonical_shape[i, 0], canonical_shape[j, 0]], 
                    [canonical_shape[i, 1], canonical_shape[j, 1]], 
                    'g-', linewidth=1.5, alpha=0.7)
    
    for i, j in mediastinal_connections:
        axes[1].plot([canonical_shape[i, 0], canonical_shape[j, 0]], 
                    [canonical_shape[i, 1], canonical_shape[j, 1]], 
                    'orange', linewidth=1.5, alpha=0.7, linestyle='--')
    
    axes[1].set_title(f"Warped to Canonical\nMorph Dist: {morph_distance:.1f}px")
    axes[1].axis('off')
    
    # Panel 3: Triangulation overlay
    axes[2].imshow(original_image, cmap='gray', alpha=0.8)
    
    # Draw triangulation
    tri = morpher.create_triangulation(predicted_landmarks, add_boundary=False)
    for simplex in tri.simplices:
        triangle = predicted_landmarks[simplex]
        triangle = np.vstack([triangle, triangle[0]])
        axes[2].plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.8, alpha=0.4)
    
    # Draw anatomical connections (stronger)
    for i, j in contour_connections:
        axes[2].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                    [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                    'g-', linewidth=2, alpha=0.9)
    
    for i, j in mediastinal_connections:
        axes[2].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                    [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                    'orange', linewidth=2, alpha=0.9, linestyle='--')
    
    axes[2].scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], 
                   c='red', s=30, edgecolors='white', linewidth=1)
    
    axes[2].set_title(f"Delaunay Triangulation\nMin Angle: {min_angle:.1f}°")
    axes[2].axis('off')
    
    # Add pathology to title
    pathology = "Unknown"
    if "COVID" in image_name:
        pathology = "COVID"
        fig.patch.set_facecolor('#ffe6e6')  # Light red
    elif "Normal" in image_name:
        pathology = "Normal"
        fig.patch.set_facecolor('#e6ffe6')  # Light green
    elif "Viral" in image_name:
        pathology = "Viral Pneumonia"
        fig.patch.set_facecolor('#e6f3ff')  # Light blue
    
    plt.suptitle(f'#{idx+1:03d} - {pathology} - ASM-Style Morphing with Template Matching', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save with pathology prefix for organization
    output_file = output_dir / f"{pathology.replace(' ', '_')}_{idx+1:03d}_{image_name}.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    return True


def main():
    """Generate all 159 visualizations."""
    print("Generating Visualizations for All 159 Processed Images")
    print("====================================================")
    
    # Load processing results
    results_path = PROJECT_ROOT / "delaunay_morphing/processed_159/processing_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load canonical shape
    canonical_shape = np.load(PROJECT_ROOT / "delaunay_morphing/processed_159/canonical_shape.npy")
    
    # Initialize morpher
    morpher = DelaunayLungMorpher()
    
    # Create output directory
    output_dir = PROJECT_ROOT / "delaunay_morphing/all_159_visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing {len(results)} images...")
    print(f"Output directory: {output_dir}")
    
    successful_results = [r for r in results if r['success']]
    print(f"Successful results to visualize: {len(successful_results)}")
    
    # Generate all visualizations
    success_count = 0
    failed_count = 0
    
    for idx, result in enumerate(tqdm(results, desc="Creating visualizations")):
        if result['success']:
            success = create_visualization_for_image(idx, result, morpher, canonical_shape, output_dir)
            if success:
                success_count += 1
            else:
                failed_count += 1
        else:
            print(f"Skipping failed result: {result['image_name']}")
            failed_count += 1
    
    print(f"\nVisualization Generation Complete:")
    print(f"  Successfully created: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(results)}")
    
    # Create index file
    create_index_html(output_dir, results)
    
    # Summary by pathology
    pathology_counts = {'COVID': 0, 'Normal': 0, 'Viral_Pneumonia': 0, 'Unknown': 0}
    
    for file in output_dir.glob("*.png"):
        if file.name.startswith("COVID"):
            pathology_counts['COVID'] += 1
        elif file.name.startswith("Normal"):
            pathology_counts['Normal'] += 1
        elif file.name.startswith("Viral"):
            pathology_counts['Viral_Pneumonia'] += 1
        else:
            pathology_counts['Unknown'] += 1
    
    print(f"\nFiles generated by pathology:")
    for pathology, count in pathology_counts.items():
        print(f"  {pathology}: {count} files")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def create_index_html(output_dir, results):
    """Create an HTML index for easy browsing."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASM-Style Morphing Results - 159 Images</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
            .image-item { border: 1px solid #ddd; padding: 10px; text-align: center; }
            .covid { background-color: #ffe6e6; }
            .normal { background-color: #e6ffe6; }
            .viral { background-color: #e6f3ff; }
            img { max-width: 100%; height: auto; }
            .stats { background-color: #f5f5f5; padding: 10px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>ASM-Style Morphing Results: 159 Test Images</h1>
        <div class="stats">
            <h3>Processing Statistics:</h3>
            <p><strong>Total Images:</strong> 159</p>
            <p><strong>Template Matching Error:</strong> 5.63 ± 1.03 pixels</p>
            <p><strong>Pipeline:</strong> Image → TM Landmarks → Delaunay Triangulation → Warp to Canonical</p>
        </div>
        
        <div class="image-grid">
    """
    
    successful_results = [r for r in results if r['success']]
    
    for idx, result in enumerate(successful_results):
        image_name = result['image_name']
        tm_error = result['template_matching_error']
        morph_distance = result['morphing_distance']
        
        # Determine pathology and class
        if "COVID" in image_name:
            pathology = "COVID"
            css_class = "covid"
            file_prefix = "COVID"
        elif "Normal" in image_name:
            pathology = "Normal"
            css_class = "normal"
            file_prefix = "Normal"
        elif "Viral" in image_name:
            pathology = "Viral Pneumonia"
            css_class = "viral"
            file_prefix = "Viral_Pneumonia"
        else:
            pathology = "Unknown"
            css_class = ""
            file_prefix = "Unknown"
        
        filename = f"{file_prefix}_{idx+1:03d}_{image_name}.png"
        
        html_content += f"""
            <div class="image-item {css_class}">
                <h4>#{idx+1}: {pathology}</h4>
                <p><strong>{image_name}</strong></p>
                <p>TM Error: {tm_error:.2f}px | Morph Dist: {morph_distance:.1f}px</p>
                <img src="{filename}" alt="{image_name}">
            </div>
        """
    
    html_content += """
        </div>
        <footer style="margin-top: 40px; text-align: center; color: #666;">
            <p>Generated by Delaunay Morphing System - Template Matching Integration</p>
        </footer>
    </body>
    </html>
    """
    
    with open(output_dir / "index.html", 'w') as f:
        f.write(html_content)
    
    print(f"✓ HTML index created: {output_dir}/index.html")


if __name__ == "__main__":
    main()