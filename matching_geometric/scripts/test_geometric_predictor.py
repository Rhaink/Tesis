#!/usr/bin/env python3
"""
Test script for GeometricLandmarkPredictor.
Tests the detection of key points 0 and 1, and generation of remaining landmarks.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add project paths
PROJECT_ROOT = '/home/donrobot/Projects/Tesis'
sys.path.append(os.path.join(PROJECT_ROOT, 'matching_geometric/src/core'))
sys.path.append(os.path.join(PROJECT_ROOT, 'template_matching/src/core'))

from geometric_predictor import GeometricLandmarkPredictor


def load_test_image(image_path: str) -> np.ndarray:
    """Load and preprocess test image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to standard size if needed
    if img.shape != (299, 299):
        img = cv2.resize(img, (299, 299))
    
    return img


def test_key_point_detection():
    """Test detection of key points 0 and 1."""
    print("Testing key point detection...")
    
    # Path to trained TM model (5.63px error model)
    model_path = os.path.join(PROJECT_ROOT, 
                             'template_matching/models/landmark_predictor.pkl')
    
    # Initialize predictor
    predictor = GeometricLandmarkPredictor(model_path)
    
    # Test image
    test_image_path = os.path.join(PROJECT_ROOT,
                                  'COVID-19_Radiography_Dataset/Normal/images/Normal-3173.png')
    
    image = load_test_image(test_image_path)
    
    # Predict landmarks
    result = predictor.predict_landmarks(image, image_name='Normal-3173')
    
    # Print results
    print(f"\nKey points detected:")
    print(f"Point 0: {result['key_points'][0]}")
    print(f"Point 1: {result['key_points'][1]}")
    
    print(f"\nMain line parameters:")
    print(f"Slope: {result['main_line']['slope']}")
    print(f"Perpendicular slope: {result['main_line']['perpendicular_slope']}")
    
    print(f"\nIntermediate points:")
    for name, point in result['intermediate_points'].items():
        print(f"{name}: {point}")
    
    # Visualize
    vis_img = predictor.visualize_predictions(image, result)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_img)
    plt.title('Geometric Landmark Prediction')
    plt.axis('off')
    
    # Save visualization
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test_geometric_prediction.png')
    cv2.imwrite(output_path, vis_img)
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    
    return result


def test_multiple_images():
    """Test on multiple images to verify consistency."""
    print("\nTesting on multiple images...")
    
    model_path = os.path.join(PROJECT_ROOT, 
                             'template_matching/models/landmark_predictor.pkl')
    predictor = GeometricLandmarkPredictor(model_path)
    
    # Test images from different categories
    test_images = [
        ('Normal', 'COVID-19_Radiography_Dataset/Normal/images/Normal-3173.png'),
        ('COVID', 'COVID-19_Radiography_Dataset/COVID/images/COVID-1652.png'),
        ('Viral Pneumonia', 'COVID-19_Radiography_Dataset/Viral Pneumonia/images/Viral Pneumonia-1334.png')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (category, rel_path) in enumerate(test_images):
        image_path = os.path.join(PROJECT_ROOT, rel_path)
        image = load_test_image(image_path)
        
        # Predict
        result = predictor.predict_landmarks(image)
        
        # Visualize
        vis_img = predictor.visualize_predictions(image, result)
        
        axes[idx].imshow(vis_img)
        axes[idx].set_title(f'{category}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = os.path.join(PROJECT_ROOT, 
                              'matching_geometric/visualizations/multi_image_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Multi-image comparison saved to: {output_path}")
    
    plt.show()


def analyze_geometric_consistency():
    """Analyze the geometric consistency of generated landmarks."""
    print("\nAnalyzing geometric consistency...")
    
    model_path = os.path.join(PROJECT_ROOT, 
                             'template_matching/models/landmark_predictor.pkl')
    predictor = GeometricLandmarkPredictor(model_path)
    
    # Test image
    test_image_path = os.path.join(PROJECT_ROOT,
                                  'COVID-19_Radiography_Dataset/Normal/images/Normal-3173.png')
    image = load_test_image(test_image_path)
    
    # Get predictions
    result = predictor.predict_landmarks(image)
    landmarks = result['landmarks']
    
    # Analyze perpendicular distances
    print("\nPerpendicular distances from main line:")
    
    # Main line points
    p0 = landmarks[0]
    p1 = landmarks[1]
    
    # Calculate distances for each landmark from main line
    # Using point-to-line distance formula
    for i in range(2, 15):
        if i in [8, 9, 10]:  # Skip intermediate points on the line
            continue
        
        # Point-to-line distance
        p = landmarks[i]
        # Vector from p0 to p1
        v = p1 - p0
        # Vector from p0 to p
        w = p - p0
        # Perpendicular distance
        c1 = np.dot(w, v)
        c2 = np.dot(v, v)
        if c2 == 0:
            dist = np.linalg.norm(w)
        else:
            b = c1 / c2
            pb = p0 + b * v
            dist = np.linalg.norm(p - pb)
        
        print(f"Landmark {i}: {dist:.2f} pixels")
    
    # Verify quartile spacing
    print("\nQuartile spacing verification:")
    quartiles = result['intermediate_points']
    
    # Distance between consecutive quartiles
    q_points = [p0, quartiles['cuarto1'], quartiles['medio'], 
                quartiles['cuarto3'], p1]
    
    for i in range(len(q_points) - 1):
        if isinstance(q_points[i], np.ndarray):
            pt1 = q_points[i]
        else:
            pt1 = np.array([q_points[i][0], q_points[i][1]])
        
        if isinstance(q_points[i+1], np.ndarray):
            pt2 = q_points[i+1]
        else:
            pt2 = np.array([q_points[i+1][0], q_points[i+1][1]])
        
        dist = np.linalg.norm(pt2 - pt1)
        print(f"Distance {i} to {i+1}: {dist:.2f} pixels")


if __name__ == "__main__":
    print("=" * 60)
    print("Geometric Landmark Predictor Test")
    print("=" * 60)
    
    # Test key point detection
    result = test_key_point_detection()
    
    # Test on multiple images
    test_multiple_images()
    
    # Analyze geometric consistency
    analyze_geometric_consistency()
    
    print("\nTest completed successfully!")