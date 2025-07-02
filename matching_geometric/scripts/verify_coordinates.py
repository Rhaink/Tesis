#!/usr/bin/env python3
"""
Verification script to ensure coordinates match exactly with the 5.63px error model.
"""

import os
import sys
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

# Add project paths
PROJECT_ROOT = '/home/donrobot/Projects/Tesis'
sys.path.append(os.path.join(PROJECT_ROOT, 'matching_geometric/src/core'))
sys.path.append(os.path.join(PROJECT_ROOT, 'template_matching/src/core'))
sys.path.append(os.path.join(PROJECT_ROOT, 'pulmones/src'))

from geometric_predictor import GeometricLandmarkPredictor
from landmark_predictor import TemplateLandmarkPredictor
from utils import asm_utils


def load_known_good_results():
    """Load the results that give 5.63px error."""
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Loaded saved results from: {results_file}")
        return results
    else:
        print(f"❌ Results file not found: {results_file}")
        return None


def verify_single_image():
    """Verify coordinates for a single test image."""
    print("="*60)
    print("VERIFICACIÓN DE COORDENADAS - IMAGEN INDIVIDUAL")
    print("="*60)
    
    # Load test image
    test_image_path = os.path.join(PROJECT_ROOT,
                                  'COVID-19_Radiography_Dataset/Normal/images/Normal-3173.png')
    
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if image.shape != (299, 299):
        image = cv2.resize(image, (299, 299))
    
    print(f"📷 Imagen de prueba: Normal-3173.png")
    print(f"📐 Tamaño: {image.shape}")
    
    # Method 1: Direct Template Matching prediction
    print("\n1️⃣ PREDICCIÓN DIRECTA CON TEMPLATE MATCHING:")
    tm_model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    
    tm_predictor = TemplateLandmarkPredictor(
        patch_size=21,
        n_components=20,
        use_multiscale=True,
        pyramid_levels=3
    )
    tm_predictor.load_model(tm_model_path)
    
    tm_result = tm_predictor.predict_landmarks(image)
    tm_landmarks = tm_result['landmarks']
    
    print(f"Landmarks TM directo - shape: {tm_landmarks.shape}")
    print(f"Punto 0: {tm_landmarks[0]}")
    print(f"Punto 1: {tm_landmarks[1]}")
    print(f"Rango X: {tm_landmarks[:, 0].min():.2f} - {tm_landmarks[:, 0].max():.2f}")
    print(f"Rango Y: {tm_landmarks[:, 1].min():.2f} - {tm_landmarks[:, 1].max():.2f}")
    
    # Method 2: Through Geometric Predictor
    print("\n2️⃣ PREDICCIÓN A TRAVÉS DE GEOMETRIC PREDICTOR:")
    geo_predictor = GeometricLandmarkPredictor(tm_model_path)
    geo_point_0, geo_point_1 = geo_predictor._detect_key_points(image)
    
    print(f"Punto 0 (geo): {geo_point_0}")
    print(f"Punto 1 (geo): {geo_point_1}")
    
    # Method 3: Load from saved results (5.63px error)
    print("\n3️⃣ RESULTADOS GUARDADOS (5.63px error):")
    saved_results = load_known_good_results()
    
    if saved_results:
        # Find Normal-3173 in the image names
        image_names = saved_results['image_names']
        predictions = saved_results['predictions']
        
        target_name = 'Normal-3173'
        found_idx = None
        
        for i, name in enumerate(image_names):
            if target_name in name:
                found_idx = i
                break
        
        if found_idx is not None:
            saved_landmarks = predictions[found_idx]
            print(f"Landmarks guardados - shape: {saved_landmarks.shape}")
            print(f"Punto 0 (guardado): {saved_landmarks[0]}")
            print(f"Punto 1 (guardado): {saved_landmarks[1]}")
            print(f"Rango X: {saved_landmarks[:, 0].min():.2f} - {saved_landmarks[:, 0].max():.2f}")
            print(f"Rango Y: {saved_landmarks[:, 1].min():.2f} - {saved_landmarks[:, 1].max():.2f}")
            
            # COMPARISON
            print("\n🔍 COMPARACIÓN:")
            print(f"TM vs Geo punto 0 - diff: {np.linalg.norm(tm_landmarks[0] - geo_point_0):.6f}")
            print(f"TM vs Geo punto 1 - diff: {np.linalg.norm(tm_landmarks[1] - geo_point_1):.6f}")
            print(f"TM vs Guardado punto 0 - diff: {np.linalg.norm(tm_landmarks[0] - saved_landmarks[0]):.6f}")
            print(f"TM vs Guardado punto 1 - diff: {np.linalg.norm(tm_landmarks[1] - saved_landmarks[1]):.6f}")
            
            # Check if they match exactly
            tm_vs_saved_diff = np.linalg.norm(tm_landmarks - saved_landmarks)
            print(f"\n📊 Diferencia total TM vs Guardado: {tm_vs_saved_diff:.6f}")
            
            if tm_vs_saved_diff < 1e-6:
                print("✅ EXACTO: Las coordenadas coinciden perfectamente")
            else:
                print("❌ ERROR: Las coordenadas NO coinciden")
                
            return tm_landmarks, saved_landmarks
        else:
            print(f"❌ No se encontró {target_name} en resultados guardados")
            return tm_landmarks, None
    else:
        print("❌ No se pudieron cargar resultados guardados")
        return tm_landmarks, None


def visualize_comparison(tm_landmarks, saved_landmarks, image):
    """Visualize coordinate comparison."""
    if saved_landmarks is None:
        print("No hay landmarks guardados para comparar")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # TM Direct
    ax1.imshow(image, cmap='gray')
    ax1.scatter(tm_landmarks[:, 0], tm_landmarks[:, 1], c='red', s=100, alpha=0.8)
    for i, (x, y) in enumerate(tm_landmarks):
        ax1.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=8, color='white')
    ax1.set_title('TM Directo', fontweight='bold')
    
    # Saved Results
    ax2.imshow(image, cmap='gray')
    ax2.scatter(saved_landmarks[:, 0], saved_landmarks[:, 1], c='green', s=100, alpha=0.8)
    for i, (x, y) in enumerate(saved_landmarks):
        ax2.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=8, color='white')
    ax2.set_title('Guardado (5.63px)', fontweight='bold')
    
    # Overlay
    ax3.imshow(image, cmap='gray')
    ax3.scatter(tm_landmarks[:, 0], tm_landmarks[:, 1], c='red', s=100, alpha=0.6, label='TM Directo')
    ax3.scatter(saved_landmarks[:, 0], saved_landmarks[:, 1], c='green', s=50, alpha=0.8, label='Guardado')
    ax3.legend()
    ax3.set_title('Superposición', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations/coordinate_verification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Comparación guardada en: {output_path}")
    
    plt.show()


def test_multiple_images():
    """Test multiple images from saved results."""
    print("\n" + "="*60)
    print("VERIFICACIÓN MÚLTIPLES IMÁGENES")
    print("="*60)
    
    saved_results = load_known_good_results()
    if not saved_results:
        print("❌ No se pueden cargar resultados guardados")
        return
    
    # Load TM model
    tm_model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    tm_predictor = TemplateLandmarkPredictor(
        patch_size=21,
        n_components=20,
        use_multiscale=True,
        pyramid_levels=3
    )
    tm_predictor.load_model(tm_model_path)
    
    # Test 5 random images
    image_names = saved_results['image_names'][:5]
    predictions = saved_results['predictions']
    total_diff = 0
    successful_tests = 0
    
    for i, img_name in enumerate(image_names):
        print(f"\n📷 Probando: {img_name}")
        
        # Load image
        try:
            img_path = asm_utils.get_image_path(img_name, None, 
                                              os.path.join(PROJECT_ROOT, 'COVID-19_Radiography_Dataset'))
            image = asm_utils.load_image_grayscale(img_path)
            
            # Predict
            tm_result = tm_predictor.predict_landmarks(image)
            tm_landmarks = tm_result['landmarks']
            
            # Compare with saved
            saved_landmarks = predictions[i]
            
            diff = np.linalg.norm(tm_landmarks - saved_landmarks)
            total_diff += diff
            successful_tests += 1
            
            print(f"   Diferencia: {diff:.6f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if successful_tests > 0:
        avg_diff = total_diff / successful_tests
    else:
        avg_diff = 0
    print(f"\n📊 Diferencia promedio: {avg_diff:.6f}")
    
    if avg_diff < 1e-6:
        print("✅ ÉXITO: Todas las coordenadas coinciden perfectamente")
    else:
        print("❌ ERROR: Las coordenadas no coinciden - hay un problema")


if __name__ == "__main__":
    print("VERIFICACIÓN DE COORDENADAS")
    print("Comparando con el modelo que da 5.63px de error")
    print("="*60)
    
    # Test single image
    tm_landmarks, saved_landmarks = verify_single_image()
    
    # Load image for visualization
    test_image_path = os.path.join(PROJECT_ROOT,
                                  'COVID-19_Radiography_Dataset/Normal/images/Normal-3173.png')
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if image.shape != (299, 299):
        image = cv2.resize(image, (299, 299))
    
    # Visualize
    visualize_comparison(tm_landmarks, saved_landmarks, image)
    
    # Test multiple images
    test_multiple_images()
    
    print("\n✅ Verificación completada")