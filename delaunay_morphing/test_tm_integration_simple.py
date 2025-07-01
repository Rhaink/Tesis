#!/usr/bin/env python3
"""
Simple test of Template Matching integration with Delaunay morphing.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Add project paths
PROJECT_ROOT = Path("/home/donrobot/Projects/Tesis")
sys.path.append(str(PROJECT_ROOT / "delaunay_morphing/src"))

from core.delaunay_lung_morpher import DelaunayLungMorpher


def main():
    print("Simple Template Matching Integration Test")
    print("=========================================\n")
    
    # Load Template Matching results
    results_path = PROJECT_ROOT / "template_matching/results/results_coordenadas_prueba_1.pkl"
    with open(results_path, 'rb') as f:
        tm_results = pickle.load(f)
    
    predictions = tm_results['predictions']
    ground_truth = tm_results['ground_truth']
    errors = tm_results['errors']
    
    print(f"Loaded {len(predictions)} predictions")
    print(f"Average error: {np.mean(errors):.2f} ± {np.std(errors):.2f} pixels")
    
    # Load CSV for image names
    csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
    df = pd.read_csv(csv_path, header=None)
    
    # Group by pathology
    pathology_data = {
        'COVID': {'landmarks': [], 'errors': [], 'indices': []},
        'Normal': {'landmarks': [], 'errors': [], 'indices': []},
        'Viral Pneumonia': {'landmarks': [], 'errors': [], 'indices': []}
    }
    
    for idx in range(min(len(predictions), len(df))):
        image_name = str(df.iloc[idx, -1])
        prediction = predictions[idx]
        error = np.mean(errors[idx])  # Use mean error per image
        
        for pathology in pathology_data.keys():
            if pathology.replace(' ', '-') in image_name:
                pathology_data[pathology]['landmarks'].append(prediction)
                pathology_data[pathology]['errors'].append(error)
                pathology_data[pathology]['indices'].append(idx)
                break
    
    # Print statistics
    print("\nResults by pathology:")
    for pathology, data in pathology_data.items():
        if data['errors']:
            mean_error = np.mean(data['errors'])
            std_error = np.std(data['errors'])
            print(f"{pathology}: {mean_error:.2f} ± {std_error:.2f} pixels ({len(data['errors'])} images)")
    
    # Find best and worst examples
    print("\nBest and worst examples:")
    all_errors = []
    for pathology, data in pathology_data.items():
        for error, idx in zip(data['errors'], data['indices']):
            all_errors.append((error, idx, pathology))
    
    all_errors.sort()
    
    best_error, best_idx, best_pathology = all_errors[0]
    worst_error, worst_idx, worst_pathology = all_errors[-1]
    
    print(f"Best: {best_error:.2f} pixels (index {best_idx}, {best_pathology})")
    print(f"Worst: {worst_error:.2f} pixels (index {worst_idx}, {worst_pathology})")
    
    # Create simple morphing visualization
    morpher = DelaunayLungMorpher()
    
    # Use best landmarks for each pathology for morphing
    print("\nCreating morphing visualization...")
    
    # Get best example from each pathology
    best_landmarks = {}
    for pathology, data in pathology_data.items():
        if data['errors']:
            best_idx_local = np.argmin(data['errors'])
            best_landmarks[pathology] = data['landmarks'][best_idx_local]
            print(f"Best {pathology}: error = {data['errors'][best_idx_local]:.2f} pixels")
    
    # Create morphing between COVID and Normal
    if 'COVID' in best_landmarks and 'Normal' in best_landmarks:
        covid_landmarks = best_landmarks['COVID']
        normal_landmarks = best_landmarks['Normal']
        
        # Create morphing sequence
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for idx, alpha in enumerate(alphas):
            # Interpolate landmarks
            current_landmarks = (1 - alpha) * covid_landmarks + alpha * normal_landmarks
            
            # Visualize landmarks with anatomical connections
            ax = axes[idx]
            
            # Plot landmarks
            ax.scatter(current_landmarks[:, 0], current_landmarks[:, 1], c='red', s=50)
            
            # Plot anatomical connections
            # Contour connections
            contour_connections = [
                (0, 12), (12, 3), (3, 5), (5, 7), (7, 14), (14, 1),
                (1, 13), (13, 6), (6, 4), (4, 2), (2, 11), (11, 0)
            ]
            for i, j in contour_connections:
                ax.plot([current_landmarks[i, 0], current_landmarks[j, 0]], 
                       [current_landmarks[i, 1], current_landmarks[j, 1]], 
                       'g-', linewidth=2)
            
            # Mediastinal connections
            mediastinal_connections = [(0, 8), (8, 9), (9, 10), (10, 1)]
            for i, j in mediastinal_connections:
                ax.plot([current_landmarks[i, 0], current_landmarks[j, 0]], 
                       [current_landmarks[i, 1], current_landmarks[j, 1]], 
                       'orange', linewidth=2, linestyle='--')
            
            ax.set_title(f'α = {alpha}')
            ax.set_xlim(0, 299)
            ax.set_ylim(299, 0)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('COVID → Normal Morphing (Template Matching Landmarks)', fontsize=16)
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'delaunay_morphing/tm_morphing_simple.png', dpi=150)
        plt.show()
        
        print("Morphing visualization saved!")
    
    print("\n✓ Simple integration test completed!")


if __name__ == "__main__":
    main()