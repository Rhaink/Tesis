#!/usr/bin/env python3
"""
Script to generate visualizations for ALL test images (159 images).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Setup paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")
sys.path.insert(0, SRC_DIR_PULMONES)

def create_all_visualizations():
    """Create visualizations for all 159 test images."""
    from utils import asm_utils
    
    print("🎨 GENERANDO VISUALIZACIONES PARA TODAS LAS IMÁGENES DE PRUEBA")
    print("=" * 60)
    
    # Load results
    results_file = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'results', 'results_coordenadas_prueba_1.pkl')
    if not os.path.exists(results_file):
        print(f"❌ No se encontraron resultados en: {results_file}")
        print("💡 Ejecuta primero: python3 template_matching/scripts/process_all_images.py --dataset coordenadas_prueba_1.csv")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"✅ Resultados cargados: {len(results['predictions'])} imágenes")
    
    # Create output directories
    base_viz_dir = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'visualizations', 'all_test_images')
    landmarks_dir = os.path.join(base_viz_dir, 'landmark_predictions')
    contours_dir = os.path.join(base_viz_dir, 'lung_contours')
    comparison_dir = os.path.join(base_viz_dir, 'side_by_side')
    
    for dir_path in [landmarks_dir, contours_dir, comparison_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Load images
    images_base_dir = os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset')
    
    # Calculate errors for sorting
    image_errors = []
    for pred, gt in zip(results['predictions'], results['ground_truth']):
        error_per_landmark = np.linalg.norm(pred - gt, axis=1)
        image_errors.append(np.mean(error_per_landmark))
    
    # Process each image
    print(f"\n🖼️ Procesando {len(results['predictions'])} imágenes...")
    
    for idx in tqdm(range(len(results['predictions'])), desc="Generando visualizaciones"):
        img_name = results['image_names'][idx]
        pred_landmarks = results['predictions'][idx]
        gt_landmarks = results['ground_truth'][idx]
        error = image_errors[idx]
        
        # Load actual image
        img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
        if not img_path or not os.path.exists(img_path):
            continue
            
        image = asm_utils.load_image_grayscale(img_path)
        if image is None:
            continue
        
        # Clean filename for saving
        clean_name = img_name.replace('/', '_').replace(' ', '_')
        
        # 1. Landmark Predictions Visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image, cmap='gray')
        
        # Ground truth (green)
        ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='lime', s=120, alpha=0.9,
                  label='Ground Truth', marker='o', edgecolors='darkgreen', linewidth=2)
        
        # Predictions (red)
        ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=120, alpha=0.9,
                  label='Predicted', marker='x', linewidth=3)
        
        # Error lines
        for j, (gt_pt, pred_pt) in enumerate(zip(gt_landmarks, pred_landmarks)):
            ax.plot([gt_pt[0], pred_pt[0]], [gt_pt[1], pred_pt[1]], 
                   'yellow', alpha=0.5, linewidth=1.5)
        
        # Add landmark numbers
        for j, (x, y) in enumerate(gt_landmarks):
            ax.annotate(f'{j}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_title(f'{img_name} - Error: {error:.2f} px', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
        
        # Save landmark visualization
        landmark_path = os.path.join(landmarks_dir, f'{idx:03d}_{clean_name}_landmarks.png')
        plt.tight_layout()
        plt.savefig(landmark_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Contour Visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image, cmap='gray')
        
        # Anatomically correct connections (from ASM)
        contour_connections = [
            (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
            (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
        ]
        midline_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
        
        # Draw predicted contour
        for start_idx, end_idx in contour_connections:
            start_pt = pred_landmarks[start_idx]
            end_pt = pred_landmarks[end_idx]
            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                   'cyan', alpha=0.8, linewidth=3)
        
        for start_idx, end_idx in midline_connections:
            start_pt = pred_landmarks[start_idx]
            end_pt = pred_landmarks[end_idx]
            ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                   'yellow', alpha=0.8, linewidth=2, linestyle='--')
        
        # Plot landmarks
        ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=150, 
                  alpha=0.9, edgecolors='white', linewidth=2)
        
        ax.set_title(f'{img_name} - Lung Contour', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Save contour visualization
        contour_path = os.path.join(contours_dir, f'{idx:03d}_{clean_name}_contour.png')
        plt.tight_layout()
        plt.savefig(contour_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Generate summary statistics
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"Total de imágenes procesadas: {len(results['predictions'])}")
    print(f"Error promedio: {np.mean(image_errors):.2f} ± {np.std(image_errors):.2f} px")
    print(f"Error mínimo: {np.min(image_errors):.2f} px")
    print(f"Error máximo: {np.max(image_errors):.2f} px")
    
    # Save summary report
    summary_path = os.path.join(base_viz_dir, 'summary_all_images.txt')
    with open(summary_path, 'w') as f:
        f.write("VISUALIZACIONES DE TODAS LAS IMÁGENES DE PRUEBA\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total de imágenes: {len(results['predictions'])}\n")
        f.write(f"Error promedio: {np.mean(image_errors):.2f} ± {np.std(image_errors):.2f} px\n")
        f.write(f"Error mediano: {np.median(image_errors):.2f} px\n")
        f.write(f"Error mínimo: {np.min(image_errors):.2f} px\n")
        f.write(f"Error máximo: {np.max(image_errors):.2f} px\n\n")
        
        # List all images with errors
        f.write("DETALLE POR IMAGEN:\n")
        f.write("-" * 50 + "\n")
        sorted_indices = np.argsort(image_errors)
        for i, idx in enumerate(sorted_indices):
            f.write(f"{i+1:3d}. {results['image_names'][idx]:30s} | {image_errors[idx]:6.2f} px\n")
    
    print(f"\n✅ VISUALIZACIONES COMPLETADAS!")
    print(f"📁 Resultados guardados en: {base_viz_dir}")
    print(f"   • {landmarks_dir} - Predicciones de landmarks")
    print(f"   • {contours_dir} - Contornos pulmonares")
    print(f"   • {summary_path} - Resumen estadístico")
    
    return base_viz_dir

if __name__ == "__main__":
    create_all_visualizations()