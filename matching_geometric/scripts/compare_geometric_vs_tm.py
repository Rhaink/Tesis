#!/usr/bin/env python3
"""
Compare geometric quartile points vs pure Template Matching points.
Analyzes precision differences between both methods.
"""

import os
import sys
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
# import seaborn as sns  # Not available

# Add project paths
PROJECT_ROOT = '/home/donrobot/Projects/Tesis'
sys.path.append(os.path.join(PROJECT_ROOT, 'matching_geometric/src/core'))
sys.path.append(os.path.join(PROJECT_ROOT, 'pulmones/src'))

from geometric_predictor import GeometricLandmarkPredictor
from utils import asm_utils


def load_tm_results():
    """Load Template Matching results and ground truth."""
    results_file = os.path.join(PROJECT_ROOT, 'template_matching/results/results_coordenadas_prueba_1.pkl')
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return results


def calculate_geometric_quartiles(predictor, image_names):
    """Calculate geometric quartiles for all images."""
    print("Calculating geometric quartiles...")
    
    geometric_quartiles = {}
    
    for image_name in tqdm(image_names, desc="Processing geometric"):
        try:
            # Load image (though we don't really need it since we use saved results)
            images_base_dir = os.path.join(PROJECT_ROOT, 'COVID-19_Radiography_Dataset')
            img_path = asm_utils.get_image_path(image_name, None, images_base_dir)
            image = asm_utils.load_image_grayscale(img_path)
            
            # Get geometric prediction
            result = predictor.predict_landmarks(image, image_name=image_name)
            
            # Extract quartile points
            quartiles = result['intermediate_points']
            geometric_quartiles[image_name] = {
                'cuarto1': quartiles['cuarto1'],
                'medio': quartiles['medio'], 
                'cuarto3': quartiles['cuarto3']
            }
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue
    
    return geometric_quartiles


def compare_methods():
    """Compare geometric vs Template Matching methods."""
    print("="*60)
    print("COMPARISON: GEOMETRIC QUARTILES vs TEMPLATE MATCHING")
    print("="*60)
    
    # Load TM results
    tm_results = load_tm_results()
    image_names = tm_results['image_names']
    tm_predictions = tm_results['predictions']
    ground_truth = tm_results['ground_truth']
    
    print(f"Loaded {len(image_names)} test images")
    
    # Setup geometric predictor
    model_path = os.path.join(PROJECT_ROOT, 'template_matching/models/landmark_predictor.pkl')
    predictor = GeometricLandmarkPredictor(model_path)
    
    # Calculate geometric quartiles
    geometric_quartiles = calculate_geometric_quartiles(predictor, image_names)
    
    # Compare quartile points (8, 9, 10) which correspond to cuarto1, medio, cuarto3
    quartile_mapping = {
        8: 'cuarto1',
        9: 'medio', 
        10: 'cuarto3'
    }
    
    comparison_results = {
        'image_name': [],
        'landmark_id': [],
        'landmark_name': [],
        'tm_error': [],
        'geometric_error': [],
        'tm_vs_geometric_diff': [],
        'geometric_better': []
    }
    
    print("\nComparing quartile points...")
    
    for i, image_name in enumerate(tqdm(image_names, desc="Comparing")):
        if image_name not in geometric_quartiles:
            continue
            
        tm_pred = tm_predictions[i]
        gt = ground_truth[i]
        geo_quartiles = geometric_quartiles[image_name]
        
        # Compare each quartile point
        for landmark_id, quartile_name in quartile_mapping.items():
            # TM error
            tm_point = tm_pred[landmark_id]
            gt_point = gt[landmark_id]
            tm_error = np.linalg.norm(tm_point - gt_point)
            
            # Geometric error
            geo_point = geo_quartiles[quartile_name]
            geo_error = np.linalg.norm(geo_point - gt_point)
            
            # Difference between methods
            tm_vs_geo_diff = np.linalg.norm(tm_point - geo_point)
            
            # Store results
            comparison_results['image_name'].append(image_name)
            comparison_results['landmark_id'].append(landmark_id)
            comparison_results['landmark_name'].append(quartile_name)
            comparison_results['tm_error'].append(tm_error)
            comparison_results['geometric_error'].append(geo_error)
            comparison_results['tm_vs_geometric_diff'].append(tm_vs_geo_diff)
            comparison_results['geometric_better'].append(geo_error < tm_error)
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_results)
    
    return df, tm_results


def analyze_results(df):
    """Analyze comparison results."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Overall statistics
    print("📊 OVERALL STATISTICS:")
    print(f"Total comparisons: {len(df)}")
    print(f"Average TM error: {df['tm_error'].mean():.3f} ± {df['tm_error'].std():.3f} pixels")
    print(f"Average Geometric error: {df['geometric_error'].mean():.3f} ± {df['geometric_error'].std():.3f} pixels")
    print(f"Average difference between methods: {df['tm_vs_geometric_diff'].mean():.3f} ± {df['tm_vs_geometric_diff'].std():.3f} pixels")
    
    # Geometric better percentage
    geometric_better_pct = (df['geometric_better'].sum() / len(df)) * 100
    print(f"Geometric method better: {geometric_better_pct:.1f}% of cases")
    
    # By landmark
    print(f"\n📍 BY LANDMARK:")
    landmark_stats = df.groupby('landmark_name').agg({
        'tm_error': ['mean', 'std'],
        'geometric_error': ['mean', 'std'],
        'tm_vs_geometric_diff': ['mean', 'std'],
        'geometric_better': 'sum'
    }).round(3)
    
    for landmark in ['cuarto1', 'medio', 'cuarto3']:
        landmark_data = df[df['landmark_name'] == landmark]
        total_cases = len(landmark_data)
        better_cases = landmark_data['geometric_better'].sum()
        better_pct = (better_cases / total_cases) * 100
        
        print(f"\n{landmark.upper()}:")
        print(f"  TM error: {landmark_data['tm_error'].mean():.3f} ± {landmark_data['tm_error'].std():.3f}")
        print(f"  Geometric error: {landmark_data['geometric_error'].mean():.3f} ± {landmark_data['geometric_error'].std():.3f}")
        print(f"  Difference: {landmark_data['tm_vs_geometric_diff'].mean():.3f} ± {landmark_data['tm_vs_geometric_diff'].std():.3f}")
        print(f"  Geometric better: {better_pct:.1f}% ({better_cases}/{total_cases})")
    
    return landmark_stats


def create_visualizations(df):
    """Create comparison visualizations."""
    print("\n📈 Creating visualizations...")
    
    # Setup
    output_dir = os.path.join(PROJECT_ROOT, 'matching_geometric/visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Simple comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Error comparison by landmark
    ax1 = axes[0, 0]
    landmarks = ['cuarto1', 'medio', 'cuarto3']
    tm_means = [df[df['landmark_name'] == lm]['tm_error'].mean() for lm in landmarks]
    geo_means = [df[df['landmark_name'] == lm]['geometric_error'].mean() for lm in landmarks]
    
    x = np.arange(len(landmarks))
    width = 0.35
    ax1.bar(x - width/2, tm_means, width, label='Template Matching', alpha=0.8)
    ax1.bar(x + width/2, geo_means, width, label='Geometric', alpha=0.8)
    ax1.set_xlabel('Landmark')
    ax1.set_ylabel('Average Error (pixels)')
    ax1.set_title('Error Comparison: TM vs Geometric')
    ax1.set_xticks(x)
    ax1.set_xticklabels(landmarks)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Difference between methods
    ax2 = axes[0, 1]
    diff_means = [df[df['landmark_name'] == lm]['tm_vs_geometric_diff'].mean() for lm in landmarks]
    ax2.bar(landmarks, diff_means, alpha=0.8, color='orange')
    ax2.set_xlabel('Landmark')
    ax2.set_ylabel('Average Difference (pixels)')
    ax2.set_title('Difference Between Methods')
    ax2.grid(True, alpha=0.3)
    
    # Error correlation
    ax3 = axes[1, 0]
    ax3.scatter(df['tm_error'], df['geometric_error'], alpha=0.6)
    max_val = max(df['tm_error'].max(), df['geometric_error'].max())
    ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Equal error line')
    ax3.set_xlabel('TM Error (pixels)')
    ax3.set_ylabel('Geometric Error (pixels)')
    ax3.set_title('Error Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Success rate by landmark
    ax4 = axes[1, 1]
    success_rates = []
    for lm in landmarks:
        lm_data = df[df['landmark_name'] == lm]
        success_rate = (lm_data['geometric_better'].sum() / len(lm_data)) * 100
        success_rates.append(success_rate)
    
    ax4.bar(landmarks, success_rates, alpha=0.8, color='green')
    ax4.set_xlabel('Landmark')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Geometric Method Success Rate')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% line')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'geometric_vs_tm_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved: {plot_path}")
    
    # Don't show plot to avoid hanging
    plt.close()
    
    # 2. Detailed statistics table
    stats_path = os.path.join(output_dir, 'geometric_vs_tm_stats.csv')
    df.to_csv(stats_path, index=False)
    print(f"📊 Detailed stats saved: {stats_path}")


def main():
    """Main comparison function."""
    # Run comparison
    df, tm_results = compare_methods()
    
    # Analyze results
    landmark_stats = analyze_results(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Summary
    print(f"\n✅ Comparison completed!")
    print(f"📊 Results show the precision difference between geometric quartiles and pure Template Matching")


if __name__ == "__main__":
    main()