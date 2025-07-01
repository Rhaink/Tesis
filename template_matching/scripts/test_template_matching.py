#!/usr/bin/env python3
"""
Simple script to test template matching landmark detection.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import logging

# Add project root to path
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_TEMPLATE = os.path.join(PROJECT_ROOT_DIR, "template_matching", "src")
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")

sys.path.append(SRC_DIR_TEMPLATE)
sys.path.insert(0, SRC_DIR_PULMONES)

# Import modules dynamically to avoid conflicts
import importlib.util

# Import template matching predictor
spec_tm = importlib.util.spec_from_file_location("landmark_predictor", os.path.join(SRC_DIR_TEMPLATE, "core", "landmark_predictor.py"))
landmark_predictor_module = importlib.util.module_from_spec(spec_tm)
spec_tm.loader.exec_module(landmark_predictor_module)
TemplateLandmarkPredictor = landmark_predictor_module.TemplateLandmarkPredictor

# Import ASM utils
spec_asm = importlib.util.spec_from_file_location("asm_utils", os.path.join(SRC_DIR_PULMONES, "utils", "asm_utils.py"))
asm_utils = importlib.util.module_from_spec(spec_asm)
spec_asm.loader.exec_module(asm_utils)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_sample_data(coordinates_file: str, images_base_dir: str, num_samples: int = 5, num_landmarks: int = 15):
    """Load a small sample of test data."""
    logging.info(f"Loading {num_samples} sample images...")
    
    try:
        shapes, image_names = asm_utils.load_landmarks(coordinates_file, num_landmarks)
        logging.info(f"Available: {len(shapes)} coordinate sets")
    except Exception as e:
        logging.error(f"Error loading coordinates: {str(e)}")
        return [], []
    
    # Take only first num_samples
    shapes = shapes[:num_samples]
    image_names = image_names[:num_samples]
    
    images = []
    landmarks_list = []
    
    for shape, img_name in zip(shapes, image_names):
        try:
            img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
            if not img_path:
                continue
                
            image = asm_utils.load_image_grayscale(img_path)
            if image is None:
                continue
            
            images.append(image)
            landmarks_list.append(shape)
            logging.info(f"Loaded: {img_name}")
            
        except Exception as e:
            logging.warning(f"Error loading {img_name}: {str(e)}")
            continue
    
    logging.info(f"Successfully loaded {len(images)} sample images")
    return images, landmarks_list


def test_template_matching(model_path: str, test_images: list, ground_truth: list):
    """Test template matching model on sample data."""
    logging.info("Loading template matching model...")
    
    try:
        predictor = TemplateLandmarkPredictor()
        predictor.load_model(model_path)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return
    
    logging.info(f"Testing on {len(test_images)} images...")
    
    errors = []
    
    for i, (image, true_landmarks) in enumerate(zip(test_images, ground_truth)):
        try:
            logging.info(f"Processing image {i+1}/{len(test_images)}...")
            
            # Predict landmarks
            result = predictor.predict_with_confidence(image)
            pred_landmarks = result['landmarks']
            confidence = result['mean_confidence']
            
            # Compute error
            error = np.mean(np.linalg.norm(pred_landmarks - true_landmarks, axis=1))
            errors.append(error)
            
            logging.info(f"  Error: {error:.2f} pixels, Confidence: {confidence:.3f}")
            
        except Exception as e:
            logging.error(f"Error processing image {i}: {str(e)}")
            continue
    
    if errors:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        min_error = np.min(errors)
        max_error = np.max(errors)
        
        print(f"\n=== TEMPLATE MATCHING RESULTS ===")
        print(f"Samples processed: {len(errors)}")
        print(f"Mean error: {mean_error:.2f} ± {std_error:.2f} pixels")
        print(f"Min error: {min_error:.2f} pixels")
        print(f"Max error: {max_error:.2f} pixels")
        
        # Show per-sample results
        print(f"\nPer-sample errors:")
        for i, error in enumerate(errors):
            print(f"  Sample {i+1}: {error:.2f} pixels")
    else:
        print("No successful predictions")


def main():
    """Main test function."""
    setup_logging()
    
    # Paths
    model_path = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'models', 'landmark_predictor_meta.pkl')
    coordinates_file = os.path.join(PROJECT_ROOT_DIR, 'coordenadas', 'coordenadas_prueba_1.csv')
    images_base_dir = os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset')
    
    # Check if model exists
    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        return
    
    # Load sample data
    images, landmarks = load_sample_data(coordinates_file, images_base_dir, num_samples=5)
    
    if len(images) == 0:
        logging.error("No test data loaded")
        return
    
    # Test template matching
    test_template_matching(model_path, images, landmarks)


if __name__ == "__main__":
    main()