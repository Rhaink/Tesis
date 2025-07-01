#!/usr/bin/env python3
"""
Quick script to process all test images and see results.
"""

import sys
import os
sys.path.insert(0, "/home/donrobot/Projects/Tesis/pulmones/src")

def quick_process_all():
    """Process all images quickly."""
    from template_matching.scripts.process_all_images import process_dataset, get_available_datasets
    
    print("🚀 PROCESAMIENTO RÁPIDO DE TODAS LAS IMÁGENES")
    print("=" * 50)
    
    # Show available datasets
    datasets = get_available_datasets()
    print("📁 Datasets disponibles:")
    for i, (filename, info) in enumerate(datasets.items(), 1):
        print(f"  {i}. {filename} - {info['description']} ({info['count']} imágenes)")
    
    # Let user choose
    try:
        choice = input("\n🔢 Selecciona dataset (número) o Enter para usar prueba: ").strip()
        if choice:
            choice_idx = int(choice) - 1
            dataset_list = list(datasets.items())
            if 0 <= choice_idx < len(dataset_list):
                selected_file, selected_info = dataset_list[choice_idx]
                coords_file = selected_info['path']
                print(f"✅ Seleccionado: {selected_file}")
            else:
                print("❌ Opción inválida, usando dataset de prueba")
                coords_file = datasets['coordenadas_prueba_1.csv']['path']
        else:
            coords_file = datasets['coordenadas_prueba_1.csv']['path']
            print("✅ Usando dataset de prueba")
            
    except (ValueError, KeyError):
        coords_file = list(datasets.values())[0]['path']
        print("✅ Usando primer dataset disponible")
    
    # Ask for max images
    try:
        max_imgs = input("🔢 Máximo número de imágenes (Enter para todas): ").strip()
        max_images = int(max_imgs) if max_imgs else None
    except ValueError:
        max_images = None
    
    print(f"\n🔄 Procesando {'todas las imágenes' if not max_images else f'{max_images} imágenes'}...")
    
    # Process
    results = process_dataset(
        coords_file, 
        max_images=max_images,
        save_predictions=True,
        create_visualizations=True
    )
    
    return results

if __name__ == "__main__":
    quick_process_all()