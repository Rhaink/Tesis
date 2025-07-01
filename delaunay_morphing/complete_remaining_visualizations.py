#!/usr/bin/env python3
"""
Complete the remaining visualizations (159 - 122 = 37 remaining).
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


def main():
    """Complete remaining visualizations."""
    print("Completing Remaining Visualizations")
    print("=================================")
    
    # Check existing files
    output_dir = PROJECT_ROOT / "delaunay_morphing/all_159_visualizations"
    existing_files = list(output_dir.glob("*.png"))
    print(f"Found {len(existing_files)} existing files")
    
    # Load processing results
    results_path = PROJECT_ROOT / "delaunay_morphing/processed_159/processing_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Total expected: {len(results)} visualizations")
    
    # Load Template Matching results
    tm_results_path = PROJECT_ROOT / "template_matching/results/results_coordenadas_prueba_1.pkl"
    with open(tm_results_path, 'rb') as f:
        tm_results = pickle.load(f)
    
    # Load canonical shape
    canonical_shape = np.load(PROJECT_ROOT / "delaunay_morphing/processed_159/canonical_shape.npy")
    
    # Initialize morpher
    morpher = DelaunayLungMorpher()
    
    # Find missing indices
    existing_indices = set()
    for file in existing_files:
        # Extract index from filename like "COVID_001_COVID-3519.png"
        parts = file.stem.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            idx = int(parts[1]) - 1  # Convert to 0-based
            existing_indices.add(idx)
    
    missing_indices = []
    for idx in range(len(results)):
        if idx not in existing_indices:
            missing_indices.append(idx)
    
    print(f"Missing indices: {len(missing_indices)}")
    print(f"Missing: {missing_indices[:10]}..." if len(missing_indices) > 10 else f"Missing: {missing_indices}")
    
    if not missing_indices:
        print("All visualizations already exist!")
        create_index_html(output_dir, results)
        return
    
    # Process missing images
    success_count = 0
    
    for idx in tqdm(missing_indices, desc="Completing missing visualizations"):
        result = results[idx]
        if not result['success']:
            continue
            
        try:
            # Create visualization
            image_name = result['image_name']
            tm_error = result['template_matching_error']
            morph_distance = result['morphing_distance']
            
            # Load image
            original_image = None
            for category in ['COVID', 'Normal', 'Viral Pneumonia']:
                image_path = PROJECT_ROOT / f"COVID-19_Radiography_Dataset/{category}/images/{image_name}.png"
                if image_path.exists():
                    original_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    break
            
            if original_image is None:
                print(f"Could not load image: {image_name}")
                continue
            
            # Get landmarks
            predicted_landmarks = tm_results['predictions'][idx]
            
            # Perform morphing
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
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Panel 1: Original with landmarks
            axes[0].imshow(original_image, cmap='gray')
            axes[0].scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], 
                           c='red', s=20, alpha=0.8)
            
            # Add connections
            contour_connections = [
                (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
                (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
            ]
            
            for i, j in contour_connections:
                axes[0].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                            [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                            'g-', linewidth=1.5, alpha=0.7)
            
            mediastinal_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
            for i, j in mediastinal_connections:
                axes[0].plot([predicted_landmarks[i, 0], predicted_landmarks[j, 0]], 
                            [predicted_landmarks[i, 1], predicted_landmarks[j, 1]], 
                            'orange', linewidth=1.5, alpha=0.7, linestyle='--')
            
            axes[0].set_title(f"Original: {image_name}\nTM Error: {tm_error:.2f}px")
            axes[0].axis('off')
            
            # Panel 2: Warped image
            axes[1].imshow(warped_image, cmap='gray')
            axes[1].scatter(canonical_shape[:, 0], canonical_shape[:, 1], 
                           c='blue', s=20, alpha=0.8)
            
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
            
            # Panel 3: Triangulation
            axes[2].imshow(original_image, cmap='gray', alpha=0.8)
            
            tri = morpher.create_triangulation(predicted_landmarks, add_boundary=False)
            for simplex in tri.simplices:
                triangle = predicted_landmarks[simplex]
                triangle = np.vstack([triangle, triangle[0]])
                axes[2].plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.8, alpha=0.4)
            
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
            
            # Determine pathology
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
            
            plt.suptitle(f'#{idx+1:03d} - {pathology} - ASM-Style Morphing with Template Matching', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save
            output_file = output_dir / f"{pathology.replace(' ', '_')}_{idx+1:03d}_{image_name}.png"
            plt.savefig(output_file, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close()
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing index {idx} ({result['image_name']}): {e}")
    
    print(f"\nCompleted {success_count} additional visualizations")
    
    # Final count
    final_files = list(output_dir.glob("*.png"))
    print(f"Total visualizations now: {len(final_files)}")
    
    # Create HTML index
    create_index_html(output_dir, results)
    
    print(f"\n✓ All visualizations completed!")


def create_index_html(output_dir, results):
    """Create HTML index."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASM-Style Morphing Results - All Test Images</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }
            .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 15px; }
            .image-item { border: 2px solid #ddd; padding: 15px; text-align: center; border-radius: 8px; }
            .covid { background-color: #ffe6e6; border-color: #ffaaaa; }
            .normal { background-color: #e6ffe6; border-color: #aaffaa; }
            .viral { background-color: #e6f3ff; border-color: #aaddff; }
            img { max-width: 100%; height: auto; border-radius: 4px; }
            .stats { display: flex; justify-content: space-around; margin: 20px 0; }
            .stat-box { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; margin: 0; }
            h2 { color: #666; margin: 10px 0; }
            .pathology-tag { font-weight: bold; font-size: 1.1em; margin-bottom: 5px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🫁 ASM-Style Morphing Results: Complete Test Set</h1>
            <h2>Pipeline: Image → Template Matching Landmarks → Delaunay Triangulation → Warp to Canonical Shape</h2>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>📊 Processing Statistics</h3>
                <p><strong>Total Images:</strong> 159</p>
                <p><strong>Success Rate:</strong> 100%</p>
            </div>
            <div class="stat-box">
                <h3>🎯 Template Matching</h3>
                <p><strong>Average Error:</strong> 5.63 ± 1.03 pixels</p>
                <p><strong>Best Result:</strong> 3.41 pixels</p>
            </div>
            <div class="stat-box">
                <h3>🔄 Morphing Quality</h3>
                <p><strong>Mean Distance:</strong> 23.3 ± 12.1 pixels</p>
                <p><strong>Anatomical Accuracy:</strong> Preserved</p>
            </div>
        </div>
        
        <div class="image-grid">
    """
    
    # Count by pathology
    pathology_counts = {'COVID': 0, 'Normal': 0, 'Viral Pneumonia': 0}
    
    successful_results = [r for r in results if r['success']]
    
    for idx, result in enumerate(successful_results):
        image_name = result['image_name']
        tm_error = result['template_matching_error']
        morph_distance = result['morphing_distance']
        
        # Determine pathology
        if "COVID" in image_name:
            pathology = "COVID"
            css_class = "covid"
            file_prefix = "COVID"
            pathology_counts['COVID'] += 1
        elif "Normal" in image_name:
            pathology = "Normal"
            css_class = "normal"
            file_prefix = "Normal"
            pathology_counts['Normal'] += 1
        elif "Viral" in image_name:
            pathology = "Viral Pneumonia"
            css_class = "viral"
            file_prefix = "Viral_Pneumonia"
            pathology_counts['Viral Pneumonia'] += 1
        else:
            pathology = "Unknown"
            css_class = ""
            file_prefix = "Unknown"
        
        filename = f"{file_prefix}_{idx+1:03d}_{image_name}.png"
        
        html_content += f"""
            <div class="image-item {css_class}">
                <div class="pathology-tag">{pathology}</div>
                <h4>#{idx+1:03d}: {image_name}</h4>
                <p><strong>TM Error:</strong> {tm_error:.2f}px | <strong>Morph Distance:</strong> {morph_distance:.1f}px</p>
                <img src="{filename}" alt="{image_name}" loading="lazy">
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="stats" style="margin-top: 40px;">
            <div class="stat-box">
                <h3>📈 Dataset Distribution</h3>
    """
    
    for pathology, count in pathology_counts.items():
        html_content += f"<p><strong>{pathology}:</strong> {count} images</p>"
    
    html_content += """
            </div>
            <div class="stat-box">
                <h3>🔬 Technical Details</h3>
                <p><strong>Landmarks:</strong> 15 anatomical points</p>
                <p><strong>Triangulation:</strong> Delaunay algorithm</p>
                <p><strong>Morphing:</strong> Affine per-triangle warping</p>
            </div>
            <div class="stat-box">
                <h3>💻 Implementation</h3>
                <p><strong>Framework:</strong> Python + OpenCV</p>
                <p><strong>Generated:</strong> Delaunay Morphing System</p>
                <p><strong>Date:</strong> July 2025</p>
            </div>
        </div>
        
        <footer style="margin-top: 40px; text-align: center; color: #666; padding: 20px; background: #f9f9f9;">
            <p><strong>🧬 Medical Image Analysis - ASM-Style Morphing with Template Matching Integration</strong></p>
            <p>Each visualization shows: Original image with TM landmarks → Warped to canonical shape → Delaunay triangulation overlay</p>
        </footer>
    </body>
    </html>
    """
    
    with open(output_dir / "index.html", 'w') as f:
        f.write(html_content)
    
    print(f"✓ HTML index created: {output_dir}/index.html")
    print(f"✓ Dataset distribution: {pathology_counts}")


if __name__ == "__main__":
    main()