#!/usr/bin/env python3
"""
Quick test without matplotlib to avoid timeout.
"""

import os
import sys
import numpy as np
import cv2

# Add project paths
PROJECT_ROOT = '/home/donrobot/Projects/Tesis'
sys.path.append(os.path.join(PROJECT_ROOT, 'matching_geometric/src/core'))

from geometric_predictor import GeometricLandmarkPredictor

def quick_test():
    # Load model
    model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    predictor = GeometricLandmarkPredictor(model_path)
    
    # Load image
    test_image_path = os.path.join(PROJECT_ROOT,
                                  'COVID-19_Radiography_Dataset/Normal/images/Normal-3173.png')
    
    image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if image.shape != (299, 299):
        image = cv2.resize(image, (299, 299))
    
    # Predict
    result = predictor.predict_landmarks(image, image_name='Normal-3173')
    
    print("Key points detected:")
    print(f"Point 0: {result['key_points'][0]}")
    print(f"Point 1: {result['key_points'][1]}")
    
    print("\nQuartile points:")
    for name, point in result['intermediate_points'].items():
        print(f"{name}: {point}")
    
    # Visualize
    vis_img = predictor.visualize_predictions(image, result)
    
    # Save
    output_path = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations/quick_test.png')
    cv2.imwrite(output_path, vis_img)
    print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    quick_test()