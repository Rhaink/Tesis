#!/usr/bin/env python3
"""
Process all 159 test images and save visualizations with geometric method.
Shows only the 5 key points: main 2 points + 3 quartile points.
"""

import os
import sys
import numpy as np
import cv2
import pickle
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = '/home/donrobot/Projects/Tesis'
sys.path.append(os.path.join(PROJECT_ROOT, 'matching_geometric/src/core'))
sys.path.append(os.path.join(PROJECT_ROOT, 'pulmones/src'))

from geometric_predictor import GeometricLandmarkPredictor
from utils import asm_utils


def load_test_images():
    """Load all 159 test images and their names."""
    print("Loading test dataset...")
    
    # Load from saved results to get exact image list
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    
    with open(results_file, 'rb') as f:
        saved_results = pickle.load(f)
    
    image_names = saved_results['image_names']
    print(f"Found {len(image_names)} test images")
    
    return image_names


def process_single_image(predictor, image_name, output_dir):
    """Process a single image and save visualization."""
    try:
        # Load image
        images_base_dir = os.path.join(PROJECT_ROOT, 'COVID-19_Radiography_Dataset')
        img_path = asm_utils.get_image_path(image_name, None, images_base_dir)
        image = asm_utils.load_image_grayscale(img_path)
        
        # Predict landmarks
        result = predictor.predict_landmarks(image, image_name=image_name)
        
        # Create visualization
        vis_img = predictor.visualize_predictions(image, result)
        
        # Save visualization
        safe_name = image_name.replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{safe_name}.png")
        cv2.imwrite(output_path, vis_img)
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def main():
    print("="*60)
    print("PROCESSING ALL 159 TEST IMAGES - GEOMETRIC METHOD")
    print("="*60)
    
    # Setup
    model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    predictor = GeometricLandmarkPredictor(model_path)
    
    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations/all_159_images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test images
    image_names = load_test_images()
    
    print(f"\nProcessing {len(image_names)} images...")
    print(f"Output directory: {output_dir}")
    
    # Process all images
    successful = 0
    failed = 0
    failed_images = []
    
    for i, image_name in enumerate(tqdm(image_names, desc="Processing")):
        success, error = process_single_image(predictor, image_name, output_dir)
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_images.append((image_name, error))
            print(f"\n❌ Failed {image_name}: {error}")
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"✅ Successful: {successful}/{len(image_names)}")
    print(f"❌ Failed: {failed}/{len(image_names)}")
    
    if failed_images:
        print(f"\nFailed images:")
        for img_name, error in failed_images:
            print(f"  - {img_name}: {error}")
    
    print(f"\n📁 Visualizations saved in: {output_dir}")
    
    # Count saved files
    saved_files = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"📊 Total files saved: {saved_files}")


if __name__ == "__main__":
    main()