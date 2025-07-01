#!/usr/bin/env python3
"""
Script to compare template matching method with ASM on test dataset.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import argparse
import logging
import pickle

# Add project paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_TEMPLATE = os.path.join(PROJECT_ROOT_DIR, "template_matching", "src")
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")

sys.path.append(SRC_DIR_TEMPLATE)
sys.path.append(SRC_DIR_PULMONES)

# Import modules dynamically to avoid conflicts
import importlib.util

# Import template matching predictor
spec_tm = importlib.util.spec_from_file_location("landmark_predictor", os.path.join(SRC_DIR_TEMPLATE, "core", "landmark_predictor.py"))
landmark_predictor_module = importlib.util.module_from_spec(spec_tm)
spec_tm.loader.exec_module(landmark_predictor_module)
TemplateLandmarkPredictor = landmark_predictor_module.TemplateLandmarkPredictor

# Import evaluation utilities
spec_eval = importlib.util.spec_from_file_location("evaluation", os.path.join(SRC_DIR_TEMPLATE, "utils", "evaluation.py"))
evaluation_module = importlib.util.module_from_spec(spec_eval)
spec_eval.loader.exec_module(evaluation_module)
MethodComparator = evaluation_module.MethodComparator

# Import ASM fitter from pulmones
sys.path.insert(0, SRC_DIR_PULMONES)
from core.asm_fitter import ASMFitter


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_test_dataset(coordinates_file: str, images_base_dir: str, num_landmarks: int = 15):
    """Load test dataset using ASM utils."""
    logging.info("Loading test dataset...")
    
    # Load coordinates using the existing ASM utils
    SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")
    sys.path.insert(0, SRC_DIR_PULMONES)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("asm_utils", os.path.join(SRC_DIR_PULMONES, "utils", "asm_utils.py"))
    asm_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asm_utils)
    
    try:
        shapes, image_names = asm_utils.load_landmarks(coordinates_file, num_landmarks)
        logging.info(f"Loaded {len(shapes)} coordinate sets")
    except Exception as e:
        logging.error(f"Error loading coordinates: {str(e)}")
        return [], [], []
    
    images = []
    landmarks_list = []
    valid_image_names = []
    
    for i, (shape, img_name) in enumerate(zip(shapes, image_names)):
        try:
            # Get image path using ASM utils
            img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
            if not img_path:
                logging.warning(f"Image not found: {img_name}")
                continue
                
            image = asm_utils.load_image_grayscale(img_path)
            if image is None:
                logging.warning(f"Could not load image: {img_path}")
                continue
            
            images.append(image)
            landmarks_list.append(shape)
            valid_image_names.append(img_name)
            
        except Exception as e:
            logging.warning(f"Error processing image {img_name}: {str(e)}")
            continue
    
    logging.info(f"Successfully loaded {len(images)} test samples")
    return images, landmarks_list, valid_image_names


def load_template_model(model_path: str) -> TemplateLandmarkPredictor:
    """Load trained template matching model."""
    logging.info(f"Loading template matching model from {model_path}")
    
    predictor = TemplateLandmarkPredictor()
    predictor.load_model(model_path)
    
    return predictor


def load_asm_model(model_path: str) -> ASMFitter:
    """Load trained ASM model."""
    logging.info(f"Loading ASM model from {model_path}")
    
    with open(model_path, 'rb') as f:
        asm_fitter = pickle.load(f)
    
    return asm_fitter


def predict_with_template_matching(predictor: TemplateLandmarkPredictor, 
                                 images: list) -> list:
    """Generate predictions using template matching."""
    logging.info("Generating template matching predictions...")
    
    predictions = []
    
    for i, image in enumerate(images):
        try:
            result = predictor.predict_with_confidence(image)
            predictions.append(result['landmarks'])
            
            if (i + 1) % 50 == 0:
                logging.info(f"Processed {i + 1}/{len(images)} images")
                
        except Exception as e:
            logging.error(f"Error predicting image {i}: {str(e)}")
            # Add dummy prediction to maintain alignment
            dummy_landmarks = np.zeros((predictor.eigenpatch_model.n_landmarks, 2))
            predictions.append(dummy_landmarks)
    
    return predictions


def predict_with_asm(asm_fitter: ASMFitter, images: list) -> list:
    """Generate predictions using ASM."""
    logging.info("Generating ASM predictions...")
    
    predictions = []
    
    for i, image in enumerate(images):
        try:
            # Initialize with mean shape
            initial_shape = asm_fitter.shape_model.mean_shape.copy()
            
            # Scale and center initial shape for image
            h, w = image.shape
            initial_shape[:, 0] = initial_shape[:, 0] * w / 64.0 + w / 2
            initial_shape[:, 1] = initial_shape[:, 1] * h / 64.0 + h / 2
            
            # Fit ASM
            final_shape, _ = asm_fitter.fit(image, initial_shape)
            predictions.append(final_shape)
            
            if (i + 1) % 50 == 0:
                logging.info(f"Processed {i + 1}/{len(images)} images")
                
        except Exception as e:
            logging.error(f"Error with ASM prediction for image {i}: {str(e)}")
            # Add dummy prediction
            dummy_landmarks = np.zeros((asm_fitter.shape_model.n_landmarks, 2))
            predictions.append(dummy_landmarks)
    
    return predictions


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare template matching with ASM')
    
    parser.add_argument('--template_model', type=str,
                       default=os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'models', 'landmark_predictor.pkl'),
                       help='Path to trained template matching model')
    
    parser.add_argument('--asm_model', type=str,
                       default=os.path.join(PROJECT_ROOT_DIR, 'pulmones', 'models', 'full_augmentation_asm_fitter.pkl'),
                       help='Path to trained ASM model')
    
    parser.add_argument('--test_coordinates', type=str,
                       default=os.path.join(PROJECT_ROOT_DIR, 'coordenadas', 'coordenadas_prueba_1.csv'),
                       help='CSV file containing test coordinates and image names')
    
    parser.add_argument('--images_base_dir', type=str,
                       default=os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset'),
                       help='Base directory containing image subdirectories')
    
    parser.add_argument('--num_landmarks', type=int, default=15,
                       help='Number of landmarks per shape')
    
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'evaluation'),
                       help='Output directory for comparison results')
    
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of test samples to use')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load test dataset
        images, ground_truth, image_names = load_test_dataset(
            args.test_coordinates, args.images_base_dir, args.num_landmarks
        )
        
        if len(images) == 0:
            logging.error("No test data loaded")
            return
        
        # Limit samples if requested
        if args.max_samples and args.max_samples < len(images):
            images = images[:args.max_samples]
            ground_truth = ground_truth[:args.max_samples]
            image_names = image_names[:args.max_samples]
            logging.info(f"Limited to {len(images)} samples")
        
        # Load models
        template_predictor = None
        asm_fitter = None
        
        template_predictions = []
        asm_predictions = []
        
        # Template matching predictions
        if os.path.exists(args.template_model):
            try:
                template_predictor = load_template_model(args.template_model)
                template_predictions = predict_with_template_matching(template_predictor, images)
            except Exception as e:
                logging.error(f"Failed to load/use template matching model: {str(e)}")
        else:
            logging.warning(f"Template matching model not found: {args.template_model}")
        
        # ASM predictions
        if os.path.exists(args.asm_model):
            try:
                asm_fitter = load_asm_model(args.asm_model)
                asm_predictions = predict_with_asm(asm_fitter, images)
            except Exception as e:
                logging.error(f"Failed to load/use ASM model: {str(e)}")
        else:
            logging.warning(f"ASM model not found: {args.asm_model}")
        
        # Compare methods
        if template_predictions and asm_predictions:
            logging.info("Comparing methods...")
            
            comparator = MethodComparator()
            comparison_results = comparator.compare_template_matching_vs_asm(
                template_predictions, asm_predictions, ground_truth,
                image_names, args.output_dir
            )
            
            # Print summary
            print("\n=== METHOD COMPARISON RESULTS ===")
            print(comparison_results['comparison_table'])
            
            print(f"\nStatistical significance test:")
            sig_test = comparison_results['significance_test']
            print(f"Test: {sig_test.get('test', 'N/A')}")
            if 'p_value' in sig_test:
                print(f"P-value: {sig_test['p_value']:.4f}")
                print(f"Significant difference: {sig_test['significant']}")
            
            print(f"\nResults saved to: {args.output_dir}")
            print("Files generated:")
            for name, path in comparison_results['output_files'].items():
                print(f"  {name}: {path}")
            
        elif template_predictions:
            logging.info("Only template matching results available")
            # Evaluate template matching only
            from utils.evaluation import LandmarkEvaluator
            evaluator = LandmarkEvaluator()
            results = evaluator.evaluate_method("Template Matching", template_predictions, ground_truth, image_names)
            
            print(f"\nTemplate Matching Results:")
            print(f"  Mean error: {results['mean_error']:.2f} pixels")
            print(f"  Std error: {results['std_error']:.2f} pixels")
            print(f"  Median error: {results['median_error']:.2f} pixels")
            
        elif asm_predictions:
            logging.info("Only ASM results available")
            # Evaluate ASM only  
            from utils.evaluation import LandmarkEvaluator
            evaluator = LandmarkEvaluator()
            results = evaluator.evaluate_method("ASM", asm_predictions, ground_truth, image_names)
            
            print(f"\nASM Results:")
            print(f"  Mean error: {results['mean_error']:.2f} pixels")
            print(f"  Std error: {results['std_error']:.2f} pixels")
            print(f"  Median error: {results['median_error']:.2f} pixels")
            
        else:
            logging.error("No valid predictions generated from either method")
        
        logging.info("Comparison completed successfully!")
        
    except Exception as e:
        logging.error(f"Comparison failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()