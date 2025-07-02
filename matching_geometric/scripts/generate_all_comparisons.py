#!/usr/bin/env python3
"""
Generar visualizaciones comparativas para las 159 imágenes de prueba.
Muestra Ground Truth vs Template Matching vs Matching Geometric.
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
    """Cargar datos de Template Matching e inicializar predictor geométrico."""
    print("Cargando datos...")
    
    # Cargar resultados TM
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    with open(results_file, 'rb') as f:
        tm_results = pickle.load(f)
    
    # Inicializar predictor geométrico
    model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    geo_predictor = GeometricLandmarkPredictor(model_path)
    
    return tm_results, geo_predictor


def draw_landmarks_minimal(image, landmarks, color, size=3):
    """Dibujar landmarks de forma minimalista."""
    img_vis = image.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    
    # Solo dibujar puntos, sin números ni líneas para mantener limpio
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_vis, (int(x), int(y)), size, color, -1)
    
    return img_vis


def draw_geometric_minimal(image, landmarks, quartiles):
    """Dibujar método geométrico: línea principal + cuartiles."""
    img_vis = image.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    
    # Línea principal (puntos 0 y 1)
    cv2.line(img_vis, 
             tuple(landmarks[0].astype(int)), 
             tuple(landmarks[1].astype(int)), 
             (0, 255, 0), 2)  # Verde
    
    # Puntos principales (0, 1)
    for i in [0, 1]:
        cv2.circle(img_vis, tuple(landmarks[i].astype(int)), 4, (0, 0, 255), -1)  # Rojo
    
    # Puntos cuartil
    for quartile_point in quartiles.values():
        cv2.circle(img_vis, tuple(quartile_point.astype(int)), 3, (0, 255, 255), -1)  # Amarillo
    
    return img_vis


def create_single_comparison_image(image, image_name, gt_landmarks, tm_landmarks, geo_predictor):
    """Crear comparación de un solo método para una imagen."""
    # Predicción geométrica
    geo_result = geo_predictor.predict_landmarks(image, image_name=image_name)
    geo_landmarks = geo_result['landmarks']
    geo_quartiles = geo_result['intermediate_points']
    
    # Crear visualizaciones
    gt_vis = draw_landmarks_minimal(image, gt_landmarks, (0, 255, 0), size=3)      # Verde
    tm_vis = draw_landmarks_minimal(image, tm_landmarks, (255, 0, 0), size=3)      # Rojo
    geo_vis = draw_geometric_minimal(image, geo_landmarks, geo_quartiles)          # Híbrido
    
    # Crear figura compacta
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Ground Truth
    axes[0].imshow(gt_vis[:, :, ::-1])  # BGR a RGB
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Template Matching
    axes[1].imshow(tm_vis[:, :, ::-1])
    axes[1].set_title('Template Matching', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Matching Geometric
    axes[2].imshow(geo_vis[:, :, ::-1])
    axes[2].set_title('Matching Geométrico', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Título general con nombre de imagen
    category = 'Normal' if 'Normal' in image_name else ('COVID' if 'COVID' in image_name else 'Viral')
    fig.suptitle(f'{category}: {image_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Espacio para título
    
    return fig


def process_all_images():
    """Procesar todas las 159 imágenes y crear comparaciones."""
    print("="*60)
    print("GENERANDO COMPARACIONES PARA 159 IMÁGENES")
    print("="*60)
    
    # Cargar datos
    tm_results, geo_predictor = load_data()
    
    image_names = tm_results['image_names']
    tm_predictions = tm_results['predictions']
    ground_truth = tm_results['ground_truth']
    
    print(f"Total de imágenes a procesar: {len(image_names)}")
    
    # Crear directorio de salida
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations/comparaciones_159')
    os.makedirs(output_dir, exist_ok=True)
    
    # Procesar imágenes
    images_base_dir = os.path.join(PROJECT_ROOT, 'COVID-19_Radiography_Dataset')
    successful = 0
    failed = 0
    
    for i, (image_name, tm_pred, gt) in enumerate(tqdm(zip(image_names, tm_predictions, ground_truth), 
                                                      desc="Procesando", total=len(image_names))):
        try:
            # Cargar imagen
            img_path = asm_utils.get_image_path(image_name, None, images_base_dir)
            image = asm_utils.load_image_grayscale(img_path)
            
            # Crear comparación
            fig = create_single_comparison_image(image, image_name, gt, tm_pred, geo_predictor)
            
            # Guardar imagen
            safe_name = image_name.replace('/', '_').replace(' ', '_').replace('-', '_')
            output_path = os.path.join(output_dir, f"comparacion_{i+1:03d}_{safe_name}.png")
            
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Cerrar para liberar memoria
            
            successful += 1
            
        except Exception as e:
            print(f"\n❌ Error procesando {image_name}: {e}")
            failed += 1
            continue
    
    # Resumen
    print(f"\n" + "="*60)
    print("RESUMEN DEL PROCESAMIENTO")
    print("="*60)
    print(f"✅ Exitosas: {successful}/{len(image_names)}")
    print(f"❌ Fallidas: {failed}/{len(image_names)}")
    print(f"📁 Directorio: {output_dir}")
    
    # Verificar archivos guardados
    saved_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"📊 Archivos guardados: {saved_files}")
    
    if saved_files > 0:
        print(f"\n📸 Ejemplos de archivos generados:")
        example_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])[:3]
        for example in example_files:
            print(f"  - {example}")
    
    return output_dir, successful, failed


def create_index_summary(output_dir, total_processed):
    """Crear un resumen índice de las comparaciones generadas."""
    print("\nCreando resumen índice...")
    
    # Contar por categoría
    files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    categories = {'Normal': 0, 'COVID': 0, 'Viral': 0}
    
    for filename in files:
        if 'Normal' in filename:
            categories['Normal'] += 1
        elif 'COVID' in filename:
            categories['COVID'] += 1
        elif 'Viral' in filename:
            categories['Viral'] += 1
    
    # Crear archivo de resumen
    summary_path = os.path.join(output_dir, 'RESUMEN.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE COMPARACIONES GENERADAS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total de comparaciones: {total_processed}\n\n")
        f.write("Por categoría:\n")
        for cat, count in categories.items():
            f.write(f"  {cat}: {count} imágenes\n")
        f.write(f"\nCada imagen muestra:\n")
        f.write(f"  - Ground Truth (referencia manual)\n")
        f.write(f"  - Template Matching (15 landmarks)\n")
        f.write(f"  - Matching Geométrico (línea + cuartiles)\n")
        f.write(f"\nUbicación: {output_dir}\n")
    
    print(f"📄 Resumen guardado: {summary_path}")
    
    # Mostrar estadísticas
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} comparaciones")


def main():
    """Función principal."""
    output_dir, successful, failed = process_all_images()
    
    if successful > 0:
        create_index_summary(output_dir, successful)
        
        print(f"\n🎉 ¡PROCESO COMPLETADO!")
        print(f"Se generaron {successful} comparaciones visuales.")
        print(f"Cada imagen muestra los 3 métodos lado a lado.")


if __name__ == "__main__":
    main()