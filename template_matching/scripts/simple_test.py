#!/usr/bin/env python3
"""
Simple test for template matching functionality.
"""

import sys
import os
import numpy as np
import cv2
import pickle
import logging

# Setup paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
sys.path.insert(0, os.path.join(PROJECT_ROOT_DIR, "template_matching", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT_DIR, "pulmones", "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_eigenpatches_model():
    """Test if eigenpatches model can be loaded and used."""
    try:
        # Import using exec
        eigenpatches_path = os.path.join(PROJECT_ROOT_DIR, "template_matching", "src", "core", "eigenpatches.py")
        with open(eigenpatches_path, 'r') as f:
            code = f.read()
        
        # Create namespace and execute
        namespace = {}
        exec(code, namespace)
        EigenpatchesModel = namespace['EigenpatchesModel']
        
        logging.info("✓ EigenpatchesModel imported successfully")
        
        # Test basic functionality
        model = EigenpatchesModel(patch_size=15, n_components=5)
        logging.info("✓ EigenpatchesModel created successfully")
        
        # Create dummy data for test
        dummy_images = [np.random.randint(0, 255, (64, 64), dtype=np.uint8) for _ in range(3)]
        dummy_landmarks = [np.random.rand(5, 2) * 64 for _ in range(3)]
        
        # Test training
        model.train(dummy_images, dummy_landmarks)
        logging.info("✓ Model training completed")
        
        # Test prediction
        test_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        predictions = model.predict_landmarks(test_image)
        logging.info(f"✓ Prediction completed: shape {predictions.shape}")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Error testing EigenpatchesModel: {str(e)}")
        return False


def test_model_loading():
    """Test loading the trained model."""
    try:
        model_path = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'models', 'single_scale_eigenpatches.pkl')
        
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logging.info(f"✓ Model loaded successfully")
        logging.info(f"  Model type: {type(model_data)}")
        if hasattr(model_data, 'patch_size'):
            logging.info(f"  Patch size: {model_data.patch_size}")
        if hasattr(model_data, 'n_components'):
            logging.info(f"  Components: {model_data.n_components}")
        if hasattr(model_data, 'n_landmarks'):
            logging.info(f"  Landmarks: {model_data.n_landmarks}")
            
        return True
        
    except Exception as e:
        logging.error(f"✗ Error loading model: {str(e)}")
        return False


def test_dataset_loading():
    """Test loading dataset using ASM utils."""
    try:
        from utils import asm_utils
        logging.info("✓ ASM utils imported successfully")
        
        coords_file = os.path.join(PROJECT_ROOT_DIR, 'coordenadas', 'coordenadas_prueba_1.csv')
        if not os.path.exists(coords_file):
            logging.warning(f"Coordinates file not found: {coords_file}")
            return False
        
        shapes, image_names = asm_utils.load_landmarks(coords_file, num_landmarks=15)
        logging.info(f"✓ Loaded {len(shapes)} shapes and {len(image_names)} image names")
        
        if len(shapes) > 0:
            logging.info(f"  First shape size: {shapes[0].shape}")
            logging.info(f"  First image name: {image_names[0]}")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Error testing dataset loading: {str(e)}")
        return False


def test_single_prediction():
    """Test making a prediction on a single image."""
    try:
        from core.eigenpatches import EigenpatchesModel
        from utils import asm_utils
        
        # Load a trained model
        model_path = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'models', 'single_scale_eigenpatches.pkl')
        if not os.path.exists(model_path):
            logging.warning("No trained model found")
            return False
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load one test image
        coords_file = os.path.join(PROJECT_ROOT_DIR, 'coordenadas', 'coordenadas_prueba_1.csv')
        shapes, image_names = asm_utils.load_landmarks(coords_file, num_landmarks=15)
        
        if len(image_names) == 0:
            logging.warning("No test images available")
            return False
        
        # Load first image
        img_name = image_names[0]
        true_landmarks = shapes[0]
        
        img_path = asm_utils.get_image_path(img_name, None, os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset'))
        if not img_path:
            logging.warning(f"Image not found: {img_name}")
            return False
        
        image = asm_utils.load_image_grayscale(img_path)
        if image is None:
            logging.warning(f"Could not load image: {img_path}")
            return False
        
        logging.info(f"✓ Loaded test image: {img_name} (shape: {image.shape})")
        
        # Make prediction
        pred_landmarks = model.predict_landmarks(image)
        logging.info(f"✓ Prediction completed")
        
        # Compute error
        error = np.mean(np.linalg.norm(pred_landmarks - true_landmarks, axis=1))
        logging.info(f"  Prediction error: {error:.2f} pixels")
        
        # Show some landmark comparisons
        logging.info("  Sample landmark comparisons (predicted vs true):")
        for i in range(min(3, len(pred_landmarks))):
            pred_x, pred_y = pred_landmarks[i]
            true_x, true_y = true_landmarks[i]
            logging.info(f"    Landmark {i}: ({pred_x:.1f}, {pred_y:.1f}) vs ({true_x:.1f}, {true_y:.1f})")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Error in single prediction test: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("=== TEMPLATE MATCHING TESTS ===\n")
    
    tests = [
        ("Basic EigenpatchesModel functionality", test_eigenpatches_model),
        ("Model loading", test_model_loading),
        ("Dataset loading", test_dataset_loading),
        ("Single prediction", test_single_prediction),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("=== SUMMARY ===")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, result in results if result)
    print(f"\nPassed: {total_passed}/{len(results)} tests")


if __name__ == "__main__":
    main()