#!/usr/bin/env python3
"""
Versión optimizada para generar comparaciones con contornos anatómicos.
Procesa las 159 imágenes más eficientemente.
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

# Configurar matplotlib para no mostrar plots
plt.ioff()


def load_data():
    """Cargar datos una sola vez."""
    print("Cargando datos...")
    
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    with open(results_file, 'rb') as f:
        tm_results = pickle.load(f)
    
    model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    geo_predictor = GeometricLandmarkPredictor(model_path)
    
    return tm_results, geo_predictor


def draw_contours_fast(image, landmarks, color, connections, size=2, line_width=1):
    """Dibujar contornos de forma optimizada."""
    img_vis = image.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    
    # Dibujar líneas
    for p1, p2 in connections:
        if p1 < len(landmarks) and p2 < len(landmarks):
            pt1 = tuple(landmarks[p1].astype(int))
            pt2 = tuple(landmarks[p2].astype(int))
            cv2.line(img_vis, pt1, pt2, color, line_width)
    
    # Dibujar puntos (sin números para rapidez)
    for x, y in landmarks:
        cv2.circle(img_vis, (int(x), int(y)), size, color, -1)
    
    return img_vis


def create_fast_comparison(image, gt_landmarks, tm_landmarks, geo_landmarks, geo_quartiles):
    """Crear comparación rápida."""
    
    # Conexiones anatómicas
    connections = [
        (0,12), (12,3), (3,5), (5,7), (7,14), (14,1), 
        (1,13), (13,6), (6,4), (4,2), (2,11), (11,0),
        (0,8), (8,9), (9,10), (10,1)
    ]
    
    # Crear visualizaciones
    gt_vis = draw_contours_fast(image, gt_landmarks, (0, 255, 0), connections, size=3, line_width=2)
    tm_vis = draw_contours_fast(image, tm_landmarks, (0, 0, 255), connections, size=2, line_width=1)
    
    # Geométrico: línea principal + cuartiles
    geo_vis = image.copy()
    if len(geo_vis.shape) == 2:
        geo_vis = cv2.cvtColor(geo_vis, cv2.COLOR_GRAY2BGR)
    
    # Línea principal
    cv2.line(geo_vis, 
             tuple(geo_landmarks[0].astype(int)), 
             tuple(geo_landmarks[1].astype(int)), 
             (0, 255, 0), 3)
    
    # Puntos principales
    for i in [0, 1]:
        cv2.circle(geo_vis, tuple(geo_landmarks[i].astype(int)), 4, (0, 0, 255), -1)
    
    # Cuartiles
    for point in geo_quartiles.values():
        cv2.circle(geo_vis, tuple(point.astype(int)), 3, (0, 255, 255), -1)
    
    return gt_vis, tm_vis, geo_vis


def process_batch(batch_data, output_dir, batch_start):
    """Procesar un lote de imágenes."""
    successful = 0
    
    for i, (image_name, tm_pred, gt, image) in enumerate(batch_data):
        try:
            # Predicción geométrica
            geo_result = geo_predictor.predict_landmarks(image, image_name=image_name)
            geo_landmarks = geo_result['landmarks']
            geo_quartiles = geo_result['intermediate_points']
            
            # Crear comparación
            gt_vis, tm_vis, geo_vis = create_fast_comparison(
                image, gt, tm_pred, geo_landmarks, geo_quartiles
            )
            
            # Crear figura simple
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(gt_vis[:, :, ::-1])
            axes[0].set_title('Ground Truth', fontsize=10)
            axes[0].axis('off')
            
            axes[1].imshow(tm_vis[:, :, ::-1])
            axes[1].set_title('Template Matching', fontsize=10)
            axes[1].axis('off')
            
            axes[2].imshow(geo_vis[:, :, ::-1])
            axes[2].set_title('Matching Geométrico', fontsize=10)
            axes[2].axis('off')
            
            # Título
            category = 'Normal' if 'Normal' in image_name else ('COVID' if 'COVID' in image_name else 'Viral')
            fig.suptitle(f'{category}: {image_name}', fontsize=12)
            
            plt.tight_layout()
            
            # Guardar
            safe_name = image_name.replace('/', '_').replace(' ', '_').replace('-', '_')
            idx = batch_start + i + 1
            output_path = os.path.join(output_dir, f"contorno_{idx:03d}_{safe_name}.png")
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError en {image_name}: {e}")
            continue
    
    return successful


def main():
    """Función principal optimizada."""
    global geo_predictor
    
    print("="*50)
    print("GENERANDO CONTORNOS OPTIMIZADO - 159 IMÁGENES")
    print("="*50)
    
    # Cargar datos
    tm_results, geo_predictor = load_data()
    
    image_names = tm_results['image_names']
    tm_predictions = tm_results['predictions']
    ground_truth = tm_results['ground_truth']
    
    # Crear directorio
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations/contornos_159')
    os.makedirs(output_dir, exist_ok=True)
    
    # Pre-cargar imágenes en lotes
    print("Cargando imágenes...")
    images_base_dir = os.path.join(PROJECT_ROOT, 'COVID-19_Radiography_Dataset')
    
    batch_size = 20
    total_successful = 0
    
    for batch_start in range(0, len(image_names), batch_size):
        batch_end = min(batch_start + batch_size, len(image_names))
        print(f"\nProcesando lote {batch_start//batch_size + 1}: {batch_start+1}-{batch_end}")
        
        # Cargar lote
        batch_data = []
        for i in range(batch_start, batch_end):
            try:
                img_path = asm_utils.get_image_path(image_names[i], None, images_base_dir)
                image = asm_utils.load_image_grayscale(img_path)
                batch_data.append((image_names[i], tm_predictions[i], ground_truth[i], image))
            except Exception as e:
                print(f"Error cargando {image_names[i]}: {e}")
                continue
        
        # Procesar lote
        if batch_data:
            batch_successful = process_batch(batch_data, output_dir, batch_start)
            total_successful += batch_successful
            print(f"Lote completado: {batch_successful}/{len(batch_data)} exitosas")
    
    # Resumen final
    print(f"\n" + "="*50)
    print("RESUMEN FINAL")
    print("="*50)
    print(f"✅ Total exitosas: {total_successful}/159")
    print(f"📁 Directorio: {output_dir}")
    
    # Verificar archivos
    saved_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"📊 Archivos guardados: {saved_files}")
    
    if saved_files > 0:
        example_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])[:3]
        print(f"\n📸 Ejemplos:")
        for example in example_files:
            print(f"  - {example}")


if __name__ == "__main__":
    main()