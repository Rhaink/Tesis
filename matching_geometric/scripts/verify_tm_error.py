#!/usr/bin/env python3
"""
Verify Template Matching error calculation to ensure we match the 5.63px reported error.
"""

import os
import pickle
import numpy as np

PROJECT_ROOT = '/home/donrobot/Projects/Tesis'

def verify_tm_error():
    """Verify the Template Matching error calculation."""
    print("="*60)
    print("VERIFYING TEMPLATE MATCHING ERROR")
    print("="*60)
    
    # Load results
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    image_names = results['image_names']
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Calculate overall error (all 15 landmarks)
    all_errors = []
    quartile_errors = []
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        # All landmarks error
        all_landmark_errors = np.linalg.norm(pred - gt, axis=1)
        img_error = np.mean(all_landmark_errors)
        all_errors.append(img_error)
        
        # Quartile landmarks only (8, 9, 10)
        quartile_landmark_errors = all_landmark_errors[[8, 9, 10]]
        quartile_error = np.mean(quartile_landmark_errors)
        quartile_errors.append(quartile_error)
    
    # Calculate statistics
    all_errors = np.array(all_errors)
    quartile_errors = np.array(quartile_errors)
    
    print(f"\n📊 ERROR ANALYSIS:")
    print(f"ALL 15 LANDMARKS:")
    print(f"  Average error: {all_errors.mean():.3f} ± {all_errors.std():.3f} pixels")
    print(f"  Min error: {all_errors.min():.3f} pixels")
    print(f"  Max error: {all_errors.max():.3f} pixels")
    print(f"  Median error: {np.median(all_errors):.3f} pixels")
    
    print(f"\nQUARTILE LANDMARKS ONLY (8, 9, 10):")
    print(f"  Average error: {quartile_errors.mean():.3f} ± {quartile_errors.std():.3f} pixels")
    print(f"  Min error: {quartile_errors.min():.3f} pixels")
    print(f"  Max error: {quartile_errors.max():.3f} pixels")
    print(f"  Median error: {np.median(quartile_errors):.3f} pixels")
    
    # Individual landmark analysis
    print(f"\n📍 BY INDIVIDUAL LANDMARK:")
    landmark_errors = []
    for landmark_id in range(15):
        errors = []
        for pred, gt in zip(predictions, ground_truth):
            error = np.linalg.norm(pred[landmark_id] - gt[landmark_id])
            errors.append(error)
        
        errors = np.array(errors)
        landmark_errors.append(errors.mean())
        
        print(f"  Landmark {landmark_id}: {errors.mean():.3f} ± {errors.std():.3f} pixels")
    
    # Focus on quartile landmarks
    print(f"\n🎯 QUARTILE LANDMARKS DETAIL:")
    quartile_names = {8: 'cuarto1', 9: 'medio', 10: 'cuarto3'}
    
    for landmark_id, name in quartile_names.items():
        errors = []
        for pred, gt in zip(predictions, ground_truth):
            error = np.linalg.norm(pred[landmark_id] - gt[landmark_id])
            errors.append(error)
        
        errors = np.array(errors)
        print(f"  {name} (landmark {landmark_id}): {errors.mean():.3f} ± {errors.std():.3f} pixels")
    
    # Check if 5.63 matches
    print(f"\n🔍 COMPARISON WITH REPORTED 5.63px:")
    if abs(all_errors.mean() - 5.63) < 0.1:
        print(f"✅ MATCH: All landmarks error ({all_errors.mean():.3f}) matches reported 5.63px")
    else:
        print(f"❌ MISMATCH: All landmarks error ({all_errors.mean():.3f}) vs reported 5.63px")
        print(f"   Difference: {abs(all_errors.mean() - 5.63):.3f} pixels")

if __name__ == "__main__":
    verify_tm_error()