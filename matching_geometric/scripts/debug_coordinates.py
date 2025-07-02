#!/usr/bin/env python3
"""
Debug script to verify coordinate systems and Template Matching predictions.
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

from landmark_predictor import TemplateLandmarkPredictor


def debug_tm_predictions():
    """Debug Template Matching predictions to check coordinate system."""
    print("Debugging Template Matching predictions...")
    
    # Load model
    model_path = os.path.join(PROJECT_ROOT, 
                             'template_matching/models/landmark_predictor.pkl')
    
    tm_predictor = TemplateLandmarkPredictor(
        patch_size=21,
        n_components=20,
        use_multiscale=True,
        pyramid_levels=3
    )
    tm_predictor.load_model(model_path)
    
    # Test image
    test_image_path = os.path.join(PROJECT_ROOT,
                                  'COVID-19_Radiography_Dataset/Normal/images/Normal-3173.png')
    
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    print(f"Image shape: {img.shape}")
    
    # Resize if needed
    if img.shape != (299, 299):
        img = cv2.resize(img, (299, 299))
        print(f"Resized to: {img.shape}")
    
    # Get predictions
    result = tm_predictor.predict_landmarks(img)
    landmarks = result['landmarks']
    
    print(f"\nNumber of landmarks: {len(landmarks)}")
    print(f"Landmarks shape: {landmarks.shape}")
    print(f"\nFirst few landmarks:")
    for i in range(min(5, len(landmarks))):
        print(f"Landmark {i}: {landmarks[i]}")
    
    # Check coordinate ranges
    print(f"\nCoordinate ranges:")
    print(f"X range: {landmarks[:, 0].min():.2f} - {landmarks[:, 0].max():.2f}")
    print(f"Y range: {landmarks[:, 1].min():.2f} - {landmarks[:, 1].max():.2f}")
    
    # Visualize all landmarks
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw all landmarks with numbers
    for i, (x, y) in enumerate(landmarks):
        x, y = int(x), int(y)
        cv2.circle(img_color, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(img_color, str(i), (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw anatomical connections
    connections = [(0,12), (12,3), (3,5), (5,7), (7,14), (14,1), 
                  (1,13), (13,6), (6,4), (4,2), (2,11), (11,0),
                  (0,8), (8,9), (9,10), (10,1)]
    
    for p1, p2 in connections:
        pt1 = tuple(landmarks[p1].astype(int))
        pt2 = tuple(landmarks[p2].astype(int))
        cv2.line(img_color, pt1, pt2, (255, 0, 0), 1)
    
    # Save visualization
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'debug_tm_landmarks.png')
    cv2.imwrite(output_path, img_color)
    print(f"\nVisualization saved to: {output_path}")
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(img_color[:, :, ::-1])  # Convert BGR to RGB
    plt.title('Template Matching Landmarks (All)')
    plt.axis('off')
    plt.show()
    
    return landmarks


def verify_points_0_and_1(landmarks):
    """Verify points 0 and 1 form the dividing line."""
    print("\n" + "="*60)
    print("Verifying points 0 and 1...")
    
    p0 = landmarks[0]
    p1 = landmarks[1]
    
    print(f"Point 0: {p0}")
    print(f"Point 1: {p1}")
    
    # Calculate line properties
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    
    print(f"\nLine properties:")
    print(f"dx: {dx:.2f}")
    print(f"dy: {dy:.2f}")
    
    if abs(dx) < 0.001:
        print("Line is vertical")
        slope = float('inf')
    else:
        slope = dy / dx
        print(f"Slope: {slope:.2f}")
    
    # Calculate distance
    distance = np.sqrt(dx**2 + dy**2)
    print(f"Distance between points: {distance:.2f} pixels")
    
    # Check if this makes sense as a dividing line
    # Points 0 and 1 should be at top and bottom of lungs
    if p0[1] < p1[1]:
        print("Point 0 is above point 1 (correct orientation)")
    else:
        print("WARNING: Point 1 is above point 0 (might need to swap)")


if __name__ == "__main__":
    print("Template Matching Coordinate Debug")
    print("="*60)
    
    # Debug TM predictions
    landmarks = debug_tm_predictions()
    
    # Verify points 0 and 1
    verify_points_0_and_1(landmarks)