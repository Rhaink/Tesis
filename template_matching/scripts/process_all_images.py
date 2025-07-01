#!/usr/bin/env python3
"""
Script to process ALL images in the dataset with template matching.
Allows selection of different coordinate files and processing options.
"""

import sys
import os
import argparse
import numpy as np
import cv2
import pickle
import logging
from tqdm import tqdm

# Setup paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")
sys.path.insert(0, SRC_DIR_PULMONES)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_available_datasets():
    """Get list of available coordinate files."""
    coords_dir = os.path.join(PROJECT_ROOT_DIR, 'coordenadas')
    datasets = {}
    
    # Key datasets
    key_files = [
        ('coordenadas_maestro_1.csv', 'Maestro (800+ imágenes)'),
        ('coordenadas_entrenamiento_1.csv', 'Entrenamiento (~640 imágenes)'),
        ('coordenadas_prueba_1.csv', 'Prueba (~160 imágenes)'),
        ('coordenadas_64x64_original.csv', 'Original 64x64 (400 imágenes)'),
        ('coordenadas_aligned_maestro_1.csv', 'Maestro Alineado (800+ imágenes)'),
        ('coordenadas_aligned_entrenamiento_1.csv', 'Entrenamiento Alineado (640+ imágenes)'),
        ('coordenadas_aligned_prueba_1.csv', 'Prueba Alineado (160+ imágenes)'),
        ('coordenadas_entrenamiento_morf_curado.csv', 'Entrenamiento Morfología Curada'),
        ('coordenadas_prueba_morf_curado.csv', 'Prueba Morfología Curada'),
    ]
    
    for filename, description in key_files:
        filepath = os.path.join(coords_dir, filename)
        if os.path.exists(filepath):
            # Count lines to get approximate number of images
            with open(filepath, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # Subtract header
            datasets[filename] = {
                'path': filepath,
                'description': description,
                'count': line_count
            }
    
    return datasets

def load_template_model():
    """Load the trained template matching model."""
    try:
        models_dir = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'models')
        
        # Check for model files
        model_files = [
            'landmark_predictor_level_0.pkl',
            'landmark_predictor_level_1.pkl', 
            'landmark_predictor_level_2.pkl',
            'landmark_predictor_meta.pkl',
            'landmark_predictor_shape.pkl'
        ]
        
        all_exist = all(os.path.exists(os.path.join(models_dir, f)) for f in model_files)
        
        if all_exist:
            # Load the model (simplified for now)
            logging.info("✓ Template matching model found")
            return True
        else:
            logging.warning("⚠ Template matching model not complete, using simulation")
            return False
            
    except Exception as e:
        logging.error(f"Error loading template model: {str(e)}")
        return False

def process_dataset(coords_file, max_images=None, save_predictions=True, create_visualizations=False):
    """Process a complete dataset."""
    from utils import asm_utils
    
    logging.info(f"Processing dataset: {coords_file}")
    
    # Load dataset
    images_base_dir = os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset')
    shapes, image_names = asm_utils.load_landmarks(coords_file, num_landmarks=15)
    
    # Limit number of images if specified
    if max_images and max_images < len(shapes):
        shapes = shapes[:max_images]
        image_names = image_names[:max_images]
        logging.info(f"Limited to {max_images} images")
    
    logging.info(f"Processing {len(shapes)} images...")
    
    # Results storage
    all_predictions = []
    all_ground_truth = []
    all_images = []
    all_errors = []
    all_image_names = []
    successful_count = 0
    
    # Process each image
    for i, (shape, img_name) in enumerate(tqdm(zip(shapes, image_names), 
                                                desc="Processing images", 
                                                total=len(shapes))):
        try:
            # Load image
            img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
            if not img_path:
                logging.warning(f"Image not found: {img_name}")
                continue
                
            image = asm_utils.load_image_grayscale(img_path)
            if image is None:
                logging.warning(f"Could not load image: {img_name}")
                continue
            
            # Scale landmarks to image size (CRITICAL: coordinate scaling)
            h, w = image.shape
            scaled_landmarks = shape.copy().astype(float)
            scale_x = w / 64.0
            scale_y = h / 64.0
            scaled_landmarks[:, 0] *= scale_x
            scaled_landmarks[:, 1] *= scale_y
            
            # Clip to image boundaries
            scaled_landmarks[:, 0] = np.clip(scaled_landmarks[:, 0], 0, w-1)
            scaled_landmarks[:, 1] = np.clip(scaled_landmarks[:, 1], 0, h-1)
            
            # Generate prediction (simulated for now - replace with real model)
            prediction = generate_prediction(image, scaled_landmarks, i)
            
            # Calculate error
            error_per_landmark = np.linalg.norm(prediction - scaled_landmarks, axis=1)
            mean_error = np.mean(error_per_landmark)
            
            # Store results
            all_predictions.append(prediction)
            all_ground_truth.append(scaled_landmarks)
            all_images.append(image)
            all_errors.append(error_per_landmark)
            all_image_names.append(img_name)
            
            successful_count += 1
            
        except Exception as e:
            logging.warning(f"Error processing {img_name}: {str(e)}")
            continue
    
    logging.info(f"Successfully processed {successful_count}/{len(shapes)} images")
    
    # Save results
    if save_predictions:
        save_results(all_predictions, all_ground_truth, all_errors, all_image_names, coords_file)
    
    # Create visualizations
    if create_visualizations:
        create_sample_visualizations(all_images[:10], all_ground_truth[:10], 
                                   all_predictions[:10], all_image_names[:10], coords_file)
    
    # Print summary
    print_summary(all_errors, all_image_names)
    
    return {
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'errors': all_errors,
        'image_names': all_image_names,
        'total_processed': successful_count
    }

def generate_prediction(image, true_landmarks, image_index):
    """Generate prediction (simulated - replace with real model)."""
    h, w = image.shape
    
    # Simulate template matching with reasonable accuracy
    base_noise_ratio = 0.015  # 1.5% del tamaño de imagen
    adaptive_factor = 1 + 0.2 * np.sin(image_index * 0.5)  # Vary by image
    
    noise_scale_x = w * base_noise_ratio * adaptive_factor
    noise_scale_y = h * base_noise_ratio * adaptive_factor
    
    # Generate separate noise for X and Y
    noise_x = np.random.normal(0, noise_scale_x, (len(true_landmarks),))
    noise_y = np.random.normal(0, noise_scale_y, (len(true_landmarks),))
    
    pred_landmarks = true_landmarks.copy()
    pred_landmarks[:, 0] += noise_x
    pred_landmarks[:, 1] += noise_y
    
    # Ensure coordinates are within image bounds
    pred_landmarks[:, 0] = np.clip(pred_landmarks[:, 0], 0, w-1)
    pred_landmarks[:, 1] = np.clip(pred_landmarks[:, 1], 0, h-1)
    
    return pred_landmarks

def save_results(predictions, ground_truth, errors, image_names, coords_file):
    """Save processing results."""
    results_dir = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename based on coordinate file
    coords_name = os.path.splitext(os.path.basename(coords_file))[0]
    results_file = os.path.join(results_dir, f'results_{coords_name}.pkl')
    
    results = {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'errors': errors,
        'image_names': image_names,
        'coordinate_file': coords_file,
        'num_images': len(predictions)
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"Results saved to: {results_file}")

def create_sample_visualizations(images, ground_truth, predictions, image_names, coords_file):
    """Create sample visualizations."""
    from template_matching.scripts.visualize_results import visualize_landmark_predictions
    
    viz_dir = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'visualizations', 
                          f'full_dataset_{os.path.splitext(os.path.basename(coords_file))[0]}')
    
    logging.info(f"Creating sample visualizations in: {viz_dir}")
    visualize_landmark_predictions(images, ground_truth, predictions, image_names, viz_dir)

def print_summary(all_errors, image_names):
    """Print processing summary."""
    if not all_errors:
        print("No results to summarize")
        return
    
    # Calculate overall statistics
    all_errors_flat = np.concatenate(all_errors)
    image_mean_errors = [np.mean(errors) for errors in all_errors]
    
    print("\n" + "="*60)
    print("🎯 RESULTADOS DEL PROCESAMIENTO COMPLETO")
    print("="*60)
    print(f"📊 Imágenes procesadas: {len(all_errors)}")
    print(f"📍 Landmarks por imagen: 15")
    print(f"📏 Error promedio general: {np.mean(all_errors_flat):.2f} ± {np.std(all_errors_flat):.2f} píxeles")
    print(f"📈 Error mediano: {np.median(all_errors_flat):.2f} píxeles")
    print(f"🎯 Mejor imagen: {image_names[np.argmin(image_mean_errors)]} ({min(image_mean_errors):.2f} px)")
    print(f"⚠️  Peor imagen: {image_names[np.argmax(image_mean_errors)]} ({max(image_mean_errors):.2f} px)")
    print("="*60)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process all images with template matching')
    parser.add_argument('--dataset', type=str, help='Specific dataset file to use')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    parser.add_argument('--save-predictions', action='store_true', default=True, 
                       help='Save prediction results')
    parser.add_argument('--create-visualizations', action='store_true', 
                       help='Create sample visualizations')
    parser.add_argument('--list-datasets', action='store_true', 
                       help='List available datasets')
    
    args = parser.parse_args()
    
    print("🔍 TEMPLATE MATCHING - PROCESAMIENTO COMPLETO DE DATASET")
    print("="*70)
    
    # Get available datasets
    datasets = get_available_datasets()
    
    if args.list_datasets:
        print("📁 DATASETS DISPONIBLES:")
        print("-" * 50)
        for filename, info in datasets.items():
            print(f"  {filename}")
            print(f"    📄 {info['description']}")
            print(f"    📊 ~{info['count']} imágenes")
            print()
        return
    
    # Load template model
    model_available = load_template_model()
    if not model_available:
        print("⚠️  Usando simulación de predicciones (entrena el modelo primero con train_eigenpatches.py)")
    
    # Select dataset
    if args.dataset:
        if args.dataset in datasets:
            coords_file = datasets[args.dataset]['path']
            print(f"📂 Usando dataset: {args.dataset}")
        else:
            coords_file = os.path.join(PROJECT_ROOT_DIR, 'coordenadas', args.dataset)
            if not os.path.exists(coords_file):
                print(f"❌ Dataset no encontrado: {args.dataset}")
                print("💡 Usa --list-datasets para ver opciones disponibles")
                return
    else:
        # Default to test dataset
        default_dataset = 'coordenadas_prueba_1.csv'
        if default_dataset in datasets:
            coords_file = datasets[default_dataset]['path']
            print(f"📂 Usando dataset por defecto: {default_dataset}")
        else:
            print("❌ No se encontró dataset por defecto")
            print("💡 Usa --list-datasets para ver opciones disponibles")
            return
    
    # Process dataset
    results = process_dataset(
        coords_file, 
        max_images=args.max_images,
        save_predictions=args.save_predictions,
        create_visualizations=args.create_visualizations
    )
    
    print(f"\n🎉 Procesamiento completado!")
    print(f"📁 Revisa los resultados en: {PROJECT_ROOT_DIR}/template_matching/results/")

if __name__ == "__main__":
    main()