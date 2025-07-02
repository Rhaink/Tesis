#!/usr/bin/env python3
"""
Generación ultra-rápida de contornos para 159 imágenes.
"""

import os
import sys
import numpy as np
import cv2
import pickle
from tqdm import tqdm

PROJECT_ROOT = '/home/donrobot/Projects/Tesis'
sys.path.append(os.path.join(PROJECT_ROOT, 'matching_geometric/src/core'))
sys.path.append(os.path.join(PROJECT_ROOT, 'pulmones/src'))

from geometric_predictor import GeometricLandmarkPredictor
from utils import asm_utils

# Configurar para rapidez
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
plt.ioff()


def create_ultra_fast_comparison(image, gt, tm, geo_result):
    """Comparación ultra rápida."""
    
    # Conexiones
    conn = [(0,12), (12,3), (3,5), (5,7), (7,14), (14,1), 
            (1,13), (13,6), (6,4), (4,2), (2,11), (11,0),
            (0,8), (8,9), (9,10), (10,1)]
    
    h, w = image.shape
    
    # GT con contornos verdes
    gt_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for p1, p2 in conn:
        cv2.line(gt_img, tuple(gt[p1].astype(int)), tuple(gt[p2].astype(int)), (0,255,0), 1)
    for x, y in gt:
        cv2.circle(gt_img, (int(x), int(y)), 2, (0,255,0), -1)
    
    # TM con contornos rojos
    tm_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for p1, p2 in conn:
        cv2.line(tm_img, tuple(tm[p1].astype(int)), tuple(tm[p2].astype(int)), (0,0,255), 1)
    for x, y in tm:
        cv2.circle(tm_img, (int(x), int(y)), 2, (0,0,255), -1)
    
    # Geométrico: línea + cuartiles
    geo_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    geo_landmarks = geo_result['landmarks']
    geo_quartiles = geo_result['intermediate_points']
    
    # Línea principal verde
    cv2.line(geo_img, tuple(geo_landmarks[0].astype(int)), tuple(geo_landmarks[1].astype(int)), (0,255,0), 2)
    # Puntos principales rojos
    cv2.circle(geo_img, tuple(geo_landmarks[0].astype(int)), 3, (0,0,255), -1)
    cv2.circle(geo_img, tuple(geo_landmarks[1].astype(int)), 3, (0,0,255), -1)
    # Cuartiles amarillos
    for point in geo_quartiles.values():
        cv2.circle(geo_img, tuple(point.astype(int)), 2, (0,255,255), -1)
    
    # Combinar en una imagen
    combined = np.hstack([gt_img, tm_img, geo_img])
    
    return combined


def main():
    print("GENERACIÓN RÁPIDA DE CONTORNOS - 159 IMÁGENES")
    print("="*50)
    
    # Cargar datos
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    with open(results_file, 'rb') as f:
        tm_results = pickle.load(f)
    
    model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    geo_predictor = GeometricLandmarkPredictor(model_path)
    
    # Preparar datos
    image_names = tm_results['image_names']
    tm_predictions = tm_results['predictions']
    ground_truth = tm_results['ground_truth']
    
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations/contornos_rapidos_159')
    os.makedirs(output_dir, exist_ok=True)
    
    images_base_dir = os.path.join(PROJECT_ROOT, 'COVID-19_Radiography_Dataset')
    
    successful = 0
    
    for i, (image_name, tm_pred, gt) in enumerate(tqdm(zip(image_names, tm_predictions, ground_truth), 
                                                      total=len(image_names), desc="Procesando")):
        try:
            # Cargar imagen
            img_path = asm_utils.get_image_path(image_name, None, images_base_dir)
            image = asm_utils.load_image_grayscale(img_path)
            
            # Predicción geométrica
            geo_result = geo_predictor.predict_landmarks(image, image_name=image_name)
            
            # Crear comparación rápida
            combined_img = create_ultra_fast_comparison(image, gt, tm_pred, geo_result)
            
            # Guardar directamente con OpenCV (más rápido)
            safe_name = image_name.replace('/', '_').replace(' ', '_').replace('-', '_')
            output_path = os.path.join(output_dir, f"contorno_{i+1:03d}_{safe_name}.png")
            cv2.imwrite(output_path, combined_img)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError en {image_name}: {e}")
            continue
    
    print(f"\n✅ Completado: {successful}/159")
    print(f"📁 Guardado en: {output_dir}")
    
    # Verificar
    saved = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"📊 Archivos: {saved}")


if __name__ == "__main__":
    main()