#!/usr/bin/env python3
"""
Training script for eigenpatches-based landmark detection model.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from typing import List, Tuple
import argparse
import logging

# Add project root to path
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_TEMPLATE = os.path.join(PROJECT_ROOT_DIR, "template_matching", "src")
sys.path.append(SRC_DIR_TEMPLATE)

from core.eigenpatches import EigenpatchesModel, MultiScaleEigenpatches
from core.landmark_predictor import TemplateLandmarkPredictor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def load_dataset(coordinates_file: str, images_base_dir: str, num_landmarks: int = 15) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load training dataset from coordinate CSV file and images.
    
    Args:
        coordinates_file: CSV file containing coordinates and image names
        images_base_dir: Base directory containing image subdirectories
        num_landmarks: Number of landmarks per shape
        
    Returns:
        Tuple of (images, landmarks_list)
    """
    logging.info("Loading dataset...")
    
    # Load coordinates using the existing ASM utils
    SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")
    sys.path.insert(0, SRC_DIR_PULMONES)  # Insert at beginning to prioritize
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("asm_utils", os.path.join(SRC_DIR_PULMONES, "utils", "asm_utils.py"))
    asm_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asm_utils)
    
    try:
        shapes, image_names = asm_utils.load_landmarks(coordinates_file, num_landmarks)
        logging.info(f"Loaded {len(shapes)} coordinate sets")
    except Exception as e:
        logging.error(f"Error loading coordinates: {str(e)}")
        return [], []
    
    images = []
    landmarks_list = []
    
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
            
        except Exception as e:
            logging.error(f"Error processing image {img_name}: {str(e)}")
            continue
    
    logging.info(f"Successfully loaded {len(images)} training samples")
    return images, landmarks_list


def train_single_scale_model(images: List[np.ndarray], landmarks_list: List[np.ndarray],
                           patch_size: int, n_components: int) -> EigenpatchesModel:
    """Train single-scale eigenpatches model."""
    logging.info("Training single-scale eigenpatches model...")
    
    model = EigenpatchesModel(patch_size=patch_size, n_components=n_components)
    model.train(images, landmarks_list)
    
    logging.info("Single-scale model training completed")
    return model


def train_multiscale_model(images: List[np.ndarray], landmarks_list: List[np.ndarray],
                          patch_size: int, n_components: int, pyramid_levels: int) -> MultiScaleEigenpatches:
    """Train multi-scale eigenpatches model."""
    logging.info("Training multi-scale eigenpatches model...")
    
    model = MultiScaleEigenpatches(
        patch_size=patch_size, 
        n_components=n_components,
        pyramid_levels=pyramid_levels
    )
    model.train(images, landmarks_list)
    
    logging.info("Multi-scale model training completed")
    return model


def train_landmark_predictor(images: List[np.ndarray], landmarks_list: List[np.ndarray],
                           patch_size: int, n_components: int, pyramid_levels: int,
                           use_multiscale: bool) -> TemplateLandmarkPredictor:
    """Train complete landmark predictor with geometric constraints."""
    logging.info("Training landmark predictor with geometric constraints...")
    
    predictor = TemplateLandmarkPredictor(
        patch_size=patch_size,
        n_components=n_components,
        use_multiscale=use_multiscale,
        pyramid_levels=pyramid_levels
    )
    predictor.train(images, landmarks_list)
    
    logging.info("Landmark predictor training completed")
    return predictor


def evaluate_model(model, test_images: List[np.ndarray], test_landmarks: List[np.ndarray]) -> dict:
    """Evaluate model performance on test set."""
    logging.info("Evaluating model performance...")
    
    errors = []
    
    for img, true_landmarks in zip(test_images, test_landmarks):
        try:
            if hasattr(model, 'predict_with_confidence'):
                # Template landmark predictor
                result = model.predict_with_confidence(img)
                pred_landmarks = result['landmarks']
            else:
                # Basic eigenpatches model
                pred_landmarks = model.predict_landmarks(img)
            
            # Compute point-to-point error
            error = np.mean(np.linalg.norm(pred_landmarks - true_landmarks, axis=1))
            errors.append(error)
            
        except Exception as e:
            logging.error(f"Error evaluating sample: {str(e)}")
            continue
    
    if errors:
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        logging.info(f"Evaluation results:")
        logging.info(f"  Mean error: {mean_error:.2f} pixels")
        logging.info(f"  Std error: {std_error:.2f} pixels")
        
        return {
            'mean_error': mean_error,
            'std_error': std_error,
            'all_errors': errors
        }
    else:
        logging.warning("No successful evaluations")
        return {'mean_error': float('inf'), 'std_error': 0, 'all_errors': []}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train eigenpatches landmark detection model')
    
    parser.add_argument('--coordinates_file', type=str, 
                       default=os.path.join(PROJECT_ROOT_DIR, 'coordenadas', 'coordenadas_entrenamiento_1.csv'),
                       help='CSV file containing coordinates and image names')
    
    parser.add_argument('--images_base_dir', type=str,
                       default=os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset'),
                       help='Base directory containing image subdirectories')
    
    parser.add_argument('--test_coordinates_file', type=str,
                       default=os.path.join(PROJECT_ROOT_DIR, 'coordenadas', 'coordenadas_prueba_1.csv'),
                       help='CSV file for test dataset coordinates')
    
    parser.add_argument('--num_landmarks', type=int, default=15,
                       help='Number of landmarks per shape')
    
    parser.add_argument('--patch_size', type=int, default=21,
                       help='Size of patches for eigenpatches')
    
    parser.add_argument('--n_components', type=int, default=20,
                       help='Number of PCA components')
    
    parser.add_argument('--pyramid_levels', type=int, default=3,
                       help='Number of pyramid levels for multi-scale')
    
    parser.add_argument('--model_type', type=str, choices=['single', 'multiscale', 'predictor'],
                       default='predictor', help='Type of model to train')
    
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'models'),
                       help='Directory to save trained models')
    
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data to use for training')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load training dataset
        images, landmarks_list = load_dataset(
            args.coordinates_file, args.images_base_dir, args.num_landmarks
        )
        
        if len(images) == 0:
            logging.error("No training data loaded")
            return
        
        # Load test dataset if available
        test_images = []
        test_landmarks = []
        
        if os.path.exists(args.test_coordinates_file):
            test_images, test_landmarks = load_dataset(
                args.test_coordinates_file, args.images_base_dir, args.num_landmarks
            )
            logging.info(f"Loaded {len(test_images)} test samples")
        
        # If no separate test set, split training data
        if not test_images:
            n_train = int(len(images) * args.train_split)
            train_images = images[:n_train]
            train_landmarks = landmarks_list[:n_train]
            test_images = images[n_train:]
            test_landmarks = landmarks_list[n_train:]
        else:
            train_images = images
            train_landmarks = landmarks_list
        
        logging.info(f"Training on {len(train_images)} samples, testing on {len(test_images)} samples")
        
        # Train model based on type
        if args.model_type == 'single':
            model = train_single_scale_model(
                train_images, train_landmarks, args.patch_size, args.n_components
            )
            model_path = os.path.join(args.output_dir, 'single_scale_eigenpatches.pkl')
            
        elif args.model_type == 'multiscale':
            model = train_multiscale_model(
                train_images, train_landmarks, args.patch_size, 
                args.n_components, args.pyramid_levels
            )
            model_path = os.path.join(args.output_dir, 'multiscale_eigenpatches.pkl')
            
        else:  # predictor
            model = train_landmark_predictor(
                train_images, train_landmarks, args.patch_size,
                args.n_components, args.pyramid_levels, use_multiscale=True
            )
            model_path = os.path.join(args.output_dir, 'landmark_predictor.pkl')
        
        # Save model
        logging.info(f"Saving model to {model_path}")
        model.save_model(model_path)
        
        # Evaluate on test set if available
        if test_images:
            eval_results = evaluate_model(model, test_images, test_landmarks)
            
            # Save evaluation results
            eval_path = model_path.replace('.pkl', '_evaluation.txt')
            with open(eval_path, 'w') as f:
                f.write(f"Model: {args.model_type}\n")
                f.write(f"Patch size: {args.patch_size}\n")
                f.write(f"PCA components: {args.n_components}\n")
                f.write(f"Pyramid levels: {args.pyramid_levels}\n")
                f.write(f"Training samples: {len(train_images)}\n")
                f.write(f"Test samples: {len(test_images)}\n")
                f.write(f"Mean error: {eval_results['mean_error']:.2f} pixels\n")
                f.write(f"Std error: {eval_results['std_error']:.2f} pixels\n")
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()