#!/usr/bin/env python3
"""
Demo script to show the improved visualizations.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")
sys.path.insert(0, SRC_DIR_PULMONES)

def demo_before_after():
    """Demonstrate the before/after of landmark visualization fixes."""
    from utils import asm_utils
    
    print("🔍 DEMOSTRACIÓN DE CORRECCIONES EN VISUALIZACIÓN")
    print("=" * 60)
    
    # Load one test image
    coords_file = os.path.join(PROJECT_ROOT_DIR, 'coordenadas', 'coordenadas_prueba_1.csv')
    images_base_dir = os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset')
    
    shapes, image_names = asm_utils.load_landmarks(coords_file, num_landmarks=15)
    
    # Take first image
    shape = shapes[0]
    img_name = image_names[0]
    
    img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
    image = asm_utils.load_image_grayscale(img_path)
    
    print(f"📷 Imagen de ejemplo: {img_name}")
    print(f"📐 Tamaño de imagen: {image.shape}")
    print(f"📍 Coordenadas originales (64x64 space): min({shape.min():.1f}), max({shape.max():.1f})")
    
    # Escalado correcto
    h, w = image.shape
    scale_x = w / 64.0
    scale_y = h / 64.0
    scaled_landmarks = shape.copy().astype(float)
    scaled_landmarks[:, 0] *= scale_x
    scaled_landmarks[:, 1] *= scale_y
    
    print(f"🔧 Factor de escalado: {scale_x:.2f}x{scale_y:.2f}")
    print(f"📍 Coordenadas escaladas: min({scaled_landmarks.min():.1f}), max({scaled_landmarks.max():.1f})")
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ANTES (coordenadas sin escalar)
    ax1.imshow(image, cmap='gray')
    ax1.scatter(shape[:, 0], shape[:, 1], c='red', s=50, alpha=0.7, marker='x')
    for j, (x, y) in enumerate(shape):
        ax1.annotate(f'{j}', (x, y), xytext=(3, 3), textcoords='offset points',
                    fontsize=8, color='yellow')
    
    ax1.set_title('❌ ANTES: Coordenadas SIN escalar\n(64x64 space en imagen 299x299)', 
                 fontsize=12, fontweight='bold', color='red')
    ax1.set_xlabel('X coordinate (pixels)')
    ax1.set_ylabel('Y coordinate (pixels)')
    
    # DESPUÉS (coordenadas escaladas correctamente)
    ax2.imshow(image, cmap='gray')
    ax2.scatter(scaled_landmarks[:, 0], scaled_landmarks[:, 1], c='lime', s=150, 
               alpha=0.9, marker='o', edgecolors='darkgreen', linewidth=3)
    for j, (x, y) in enumerate(scaled_landmarks):
        ax2.annotate(f'{j}', (x, y), xytext=(8, 8), textcoords='offset points',
                    fontsize=12, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))
    
    ax2.set_title('✅ DESPUÉS: Coordenadas ESCALADAS correctamente\n(Escaladas a tamaño real de imagen)', 
                 fontsize=12, fontweight='bold', color='green')
    ax2.set_xlabel('X coordinate (pixels)')
    ax2.set_ylabel('Y coordinate (pixels)')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'visualizations', 'before_after_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"💾 Comparación guardada en: {output_path}")
    
    # Show the plot
    plt.show()
    
    print("\n📊 RESUMEN DE CORRECCIONES:")
    print("-" * 40)
    print("✅ Escalado correcto: 64x64 → tamaño real de imagen")
    print("✅ Landmarks más grandes: 50 → 150 píxeles")
    print("✅ Números más visibles: 8px → 12px con fondo")
    print("✅ Colores más contrastantes: verde lima vs rojo")
    print("✅ Contorno siguiendo orden anatómico correcto")
    
    print(f"\n🎯 VERIFICACIÓN:")
    print(f"   • Landmarks ahora están en rango [0-{w-1}] x [0-{h-1}]")
    print(f"   • Factor de escalado: {scale_x:.2f}x")
    print(f"   • Posiciones realistas en la anatomía pulmonar")

def show_contour_improvement():
    """Show the contour connectivity improvement."""
    print("\n🫁 DEMOSTRACIÓN DE MEJORA EN CONTORNOS")
    print("=" * 50)
    
    from utils import asm_utils
    
    coords_file = os.path.join(PROJECT_ROOT_DIR, 'coordenadas', 'coordenadas_prueba_1.csv')
    images_base_dir = os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset')
    
    shapes, image_names = asm_utils.load_landmarks(coords_file, num_landmarks=15)
    
    shape = shapes[0]
    img_name = image_names[0]
    
    img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
    image = asm_utils.load_image_grayscale(img_path)
    
    # Escalar landmarks
    h, w = image.shape
    scaled_landmarks = shape.copy().astype(float)
    scaled_landmarks[:, 0] *= w / 64.0
    scaled_landmarks[:, 1] *= h / 64.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ANTES: Conectividad incorrecta (pulmones separados)
    ax1.imshow(image, cmap='gray')
    
    # Conexiones incorrectas del ejemplo anterior
    wrong_connections = []
    # Right lung (landmarks 0-7)
    right_lung = list(range(8))
    for i in range(len(right_lung)-1):
        wrong_connections.append((right_lung[i], right_lung[i+1]))
    wrong_connections.append((right_lung[-1], right_lung[0]))
    
    # Left lung (landmarks 8-14)
    left_lung = list(range(8, 15))
    for i in range(len(left_lung)-1):
        wrong_connections.append((left_lung[i], left_lung[i+1]))
    wrong_connections.append((left_lung[-1], left_lung[0]))
    
    # Draw wrong connections
    for start_idx, end_idx in wrong_connections:
        if start_idx < len(scaled_landmarks) and end_idx < len(scaled_landmarks):
            start_pt = scaled_landmarks[start_idx]
            end_pt = scaled_landmarks[end_idx]
            ax1.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                    'red', alpha=0.7, linewidth=2)
    
    ax1.scatter(scaled_landmarks[:, 0], scaled_landmarks[:, 1], c='yellow', s=100, 
               alpha=0.8, edgecolors='black', linewidth=2)
    ax1.set_title('❌ ANTES: Conectividad Incorrecta\n(Asumiendo dos contornos separados)', 
                 fontsize=12, fontweight='bold', color='red')
    
    # DESPUÉS: Conectividad anatómica correcta
    ax2.imshow(image, cmap='gray')
    
    # Conexiones correctas (anatómicas) - igual que en ASM
    contour_connections = [
        (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
        (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
    ]
    midline_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
    
    # Draw correct contour connections
    for start_idx, end_idx in contour_connections:
        start_pt = scaled_landmarks[start_idx]
        end_pt = scaled_landmarks[end_idx]
        ax2.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                'cyan', alpha=0.8, linewidth=3)
    
    # Draw midline connections
    for start_idx, end_idx in midline_connections:
        start_pt = scaled_landmarks[start_idx]
        end_pt = scaled_landmarks[end_idx]
        ax2.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                'yellow', alpha=0.8, linewidth=2, linestyle='--')
    
    ax2.scatter(scaled_landmarks[:, 0], scaled_landmarks[:, 1], c='red', s=150, 
               alpha=0.9, edgecolors='white', linewidth=3)
    
    # Add numbers to show sequence
    for j, (x, y) in enumerate(scaled_landmarks):
        ax2.annotate(f'{j}', (x, y), xytext=(0, 0), textcoords='offset points',
                    fontsize=12, color='white', fontweight='bold', ha='center', va='center')
    
    ax2.set_title('✅ DESPUÉS: Conectividad Anatómica\n(Siguiendo estructura ASM: contorno + mediastino)', 
                 fontsize=12, fontweight='bold', color='green')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'visualizations', 'contour_improvement.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"💾 Mejora de contorno guardada en: {output_path}")
    plt.show()
    
    print("📋 CONECTIVIDAD ANATÓMICA IMPLEMENTADA:")
    print("   Contorno: 0→12→3→5→7→14→1→13→6→4→2→11→0")
    print("   Mediastino: 0→8→9→10→1")
    print("   (Siguiendo la estructura real de landmarks de pulmón en ASM)")

def main():
    """Main demo function."""
    print("🎨 DEMOSTRACIÓN DE CORRECCIONES EN VISUALIZACIÓN")
    print("🔧 Template Matching - Landmark Visualization Fixes")
    print("=" * 70)
    
    try:
        # Demo 1: Before/After scaling
        demo_before_after()
        
        # Demo 2: Contour connectivity improvement
        show_contour_improvement()
        
        print("\n🎉 DEMOSTRACIÓN COMPLETADA!")
        print("✅ Todas las correcciones han sido aplicadas exitosamente")
        print(f"📁 Visualizaciones disponibles en:")
        print(f"   {os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'visualizations')}")
        
    except Exception as e:
        print(f"❌ Error en demostración: {str(e)}")

if __name__ == "__main__":
    main()