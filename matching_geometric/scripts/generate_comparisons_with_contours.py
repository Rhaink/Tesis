#!/usr/bin/env python3
"""
Generar comparaciones con contornos anatómicos para las 159 imágenes de prueba.
Muestra Ground Truth vs Template Matching vs Matching Geometric con contornos conectados.
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


def draw_landmarks_with_contours(image, landmarks, color, size=3, line_width=1):
    """Dibujar landmarks con contornos anatómicos correctos."""
    img_vis = image.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    
    # Conexiones anatómicas correctas (del CLAUDE.md)
    connections = [
        (0,12), (12,3), (3,5), (5,7), (7,14), (14,1), 
        (1,13), (13,6), (6,4), (4,2), (2,11), (11,0),
        (0,8), (8,9), (9,10), (10,1)
    ]
    
    # Dibujar líneas de contorno
    for p1, p2 in connections:
        if p1 < len(landmarks) and p2 < len(landmarks):
            pt1 = tuple(landmarks[p1].astype(int))
            pt2 = tuple(landmarks[p2].astype(int))
            cv2.line(img_vis, pt1, pt2, color, line_width)
    
    # Dibujar puntos landmarks
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_vis, (int(x), int(y)), size, color, -1)
        # Números de landmarks (más pequeños para no sobrecargar)
        cv2.putText(img_vis, str(i), (int(x)+3, int(y)-3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    return img_vis


def draw_geometric_with_contours(image, landmarks, quartiles):
    """Dibujar método geométrico con línea principal y puntos cuartil."""
    img_vis = image.copy()
    if len(img_vis.shape) == 2:
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
    
    # Línea principal (puntos 0 y 1) - más gruesa
    cv2.line(img_vis, 
             tuple(landmarks[0].astype(int)), 
             tuple(landmarks[1].astype(int)), 
             (0, 255, 0), 3)  # Verde grueso
    
    # Puntos principales (0, 1) - más grandes
    for i in [0, 1]:
        cv2.circle(img_vis, tuple(landmarks[i].astype(int)), 5, (0, 0, 255), -1)  # Rojo
        cv2.putText(img_vis, str(i), 
                   (int(landmarks[i][0])+6, int(landmarks[i][1])-6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Puntos cuartil con etiquetas
    quartile_labels = {'cuarto1': 'Q1', 'medio': 'M', 'cuarto3': 'Q3'}
    for name, point in quartiles.items():
        cv2.circle(img_vis, tuple(point.astype(int)), 4, (0, 255, 255), -1)  # Amarillo
        if name in quartile_labels:
            cv2.putText(img_vis, quartile_labels[name], 
                       (int(point[0])+5, int(point[1])-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return img_vis


def create_contour_comparison(image, image_name, gt_landmarks, tm_landmarks, geo_predictor):
    """Crear comparación con contornos para una imagen."""
    # Predicción geométrica
    geo_result = geo_predictor.predict_landmarks(image, image_name=image_name)
    geo_landmarks = geo_result['landmarks']
    geo_quartiles = geo_result['intermediate_points']
    
    # Crear visualizaciones con contornos
    gt_vis = draw_landmarks_with_contours(image, gt_landmarks, (0, 255, 0), size=4, line_width=2)      # Verde
    tm_vis = draw_landmarks_with_contours(image, tm_landmarks, (0, 0, 255), size=3, line_width=1)      # Rojo
    geo_vis = draw_geometric_with_contours(image, geo_landmarks, geo_quartiles)                        # Híbrido
    
    # Crear figura compacta
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground Truth
    axes[0].imshow(gt_vis[:, :, ::-1])  # BGR a RGB
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold', color='green')
    axes[0].axis('off')
    
    # Template Matching
    axes[1].imshow(tm_vis[:, :, ::-1])
    axes[1].set_title('Template Matching', fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')
    
    # Matching Geometric
    axes[2].imshow(geo_vis[:, :, ::-1])
    axes[2].set_title('Matching Geométrico', fontsize=14, fontweight='bold', color='blue')
    axes[2].axis('off')
    
    # Título general con nombre de imagen
    category = 'Normal' if 'Normal' in image_name else ('COVID' if 'COVID' in image_name else 'Viral')
    fig.suptitle(f'{category}: {image_name}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Espacio para título
    
    return fig


def process_all_images_with_contours():
    """Procesar todas las 159 imágenes con contornos anatómicos."""
    print("="*60)
    print("GENERANDO COMPARACIONES CON CONTORNOS - 159 IMÁGENES")
    print("="*60)
    
    # Cargar datos
    tm_results, geo_predictor = load_data()
    
    image_names = tm_results['image_names']
    tm_predictions = tm_results['predictions']
    ground_truth = tm_results['ground_truth']
    
    print(f"Total de imágenes a procesar: {len(image_names)}")
    
    # Crear directorio de salida
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations/comparaciones_contornos_159')
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
            
            # Crear comparación con contornos
            fig = create_contour_comparison(image, image_name, gt, tm_pred, geo_predictor)
            
            # Guardar imagen
            safe_name = image_name.replace('/', '_').replace(' ', '_').replace('-', '_')
            output_path = os.path.join(output_dir, f"contorno_{i+1:03d}_{safe_name}.png")
            
            plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Cerrar para liberar memoria
            
            successful += 1
            
        except Exception as e:
            print(f"\n❌ Error procesando {image_name}: {e}")
            failed += 1
            continue
    
    # Resumen
    print(f"\n" + "="*60)
    print("RESUMEN DEL PROCESAMIENTO CON CONTORNOS")
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


def create_contour_summary(output_dir, total_processed):
    """Crear resumen de las comparaciones con contornos."""
    print("\nCreando resumen de contornos...")
    
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
    summary_path = os.path.join(output_dir, 'RESUMEN_CONTORNOS.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE COMPARACIONES CON CONTORNOS ANATÓMICOS\n")
        f.write("="*55 + "\n\n")
        f.write(f"Total de comparaciones: {total_processed}\n\n")
        f.write("Por categoría:\n")
        for cat, count in categories.items():
            f.write(f"  {cat}: {count} imágenes\n")
        f.write(f"\nCada imagen muestra:\n")
        f.write(f"  - Ground Truth: Contornos verdes con landmarks numerados\n")
        f.write(f"  - Template Matching: Contornos rojos (15 landmarks)\n")
        f.write(f"  - Matching Geométrico: Línea principal + cuartiles (Q1, M, Q3)\n")
        f.write(f"\nConexiones anatómicas:\n")
        f.write(f"  - Contorno pulmonar: (0,12), (12,3), (3,5), (5,7), (7,14), (14,1)\n")
        f.write(f"  - Contorno izquierdo: (1,13), (13,6), (6,4), (4,2), (2,11), (11,0)\n") 
        f.write(f"  - Línea mediastinal: (0,8), (8,9), (9,10), (10,1)\n")
        f.write(f"\nUbicación: {output_dir}\n")
    
    print(f"📄 Resumen de contornos guardado: {summary_path}")
    
    # Mostrar estadísticas
    print(f"\n📊 ESTADÍSTICAS FINALES CON CONTORNOS:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} comparaciones con contornos")


def main():
    """Función principal para generar comparaciones con contornos."""
    output_dir, successful, failed = process_all_images_with_contours()
    
    if successful > 0:
        create_contour_summary(output_dir, successful)
        
        print(f"\n🎉 ¡PROCESO CON CONTORNOS COMPLETADO!")
        print(f"Se generaron {successful} comparaciones con contornos anatómicos.")
        print(f"Cada imagen muestra los 3 métodos con conexiones de landmarks.")
        print(f"\n🔗 Características:")
        print(f"  ✅ Contornos anatómicos correctos")
        print(f"  ✅ Landmarks numerados")
        print(f"  ✅ Líneas de conexión por método")
        print(f"  ✅ Cuartiles etiquetados (Q1, M, Q3)")


if __name__ == "__main__":
    main()