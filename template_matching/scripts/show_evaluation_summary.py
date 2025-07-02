#!/usr/bin/env python3
"""
Display key findings from the Template Matching per-landmark evaluation.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"

def display_summary():
    """Display a concise summary of the evaluation results."""
    evaluation_dir = Path(PROJECT_ROOT_DIR) / "template_matching" / "evaluation"
    
    # Load the statistics
    overall_stats = pd.read_csv(evaluation_dir / "per_landmark_statistics.csv")
    pathology_stats = pd.read_csv(evaluation_dir / "per_landmark_by_pathology.csv")
    
    print("=" * 80)
    print("TEMPLATE MATCHING - PER-LANDMARK PRECISION EVALUATION SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 50)
    overall_mean = overall_stats['mean_error'].mean()
    overall_std = overall_stats['mean_error'].std()
    print(f"Average error across all landmarks: {overall_mean:.3f} ± {overall_std:.3f} pixels")
    print(f"Total landmarks evaluated: {len(overall_stats)}")
    print(f"Total test samples: {overall_stats['n_samples'].iloc[0]}")
    
    # Best and worst landmarks
    print("\n🎯 LANDMARK PERFORMANCE RANKING")
    print("-" * 50)
    sorted_landmarks = overall_stats.sort_values('mean_error')
    
    print("Best performing landmarks (lowest error):")
    for i, (_, landmark) in enumerate(sorted_landmarks.head(3).iterrows()):
        print(f"  {i+1}. Landmark {landmark['landmark_id']:2d}: {landmark['mean_error']:.3f} px ({landmark['anatomical_region']})")
    
    print("\nWorst performing landmarks (highest error):")
    for i, (_, landmark) in enumerate(sorted_landmarks.tail(3).iterrows()):
        print(f"  {i+1}. Landmark {landmark['landmark_id']:2d}: {landmark['mean_error']:.3f} px ({landmark['anatomical_region']})")
    
    # Pathology comparison
    print("\n🏥 PATHOLOGY BREAKDOWN")
    print("-" * 50)
    pathology_summary = pathology_stats.groupby('pathology').agg({
        'mean_error': ['mean', 'std'],
        'n_samples': 'first'
    }).round(3)
    
    for pathology in pathology_summary.index:
        mean_err = pathology_summary.loc[pathology, ('mean_error', 'mean')]
        std_err = pathology_summary.loc[pathology, ('mean_error', 'std')]
        n_samples = pathology_summary.loc[pathology, ('n_samples', 'first')]
        print(f"{pathology:15s} (n={n_samples:2d}): {mean_err:.3f} ± {std_err:.3f} pixels")
    
    # Variability analysis
    print("\n📈 VARIABILITY ANALYSIS")
    print("-" * 50)
    most_consistent = overall_stats.loc[overall_stats['std_error'].idxmin()]
    most_variable = overall_stats.loc[overall_stats['std_error'].idxmax()]
    
    print(f"Most consistent landmark: L{most_consistent['landmark_id']} (std: {most_consistent['std_error']:.3f} px)")
    print(f"Most variable landmark:   L{most_variable['landmark_id']} (std: {most_variable['std_error']:.3f} px)")
    
    # Range analysis
    overall_stats['error_range'] = overall_stats['max_error'] - overall_stats['min_error']
    smallest_range = overall_stats.loc[overall_stats['error_range'].idxmin()]
    largest_range = overall_stats.loc[overall_stats['error_range'].idxmax()]
    
    print(f"Smallest error range:     L{smallest_range['landmark_id']} ({smallest_range['error_range']:.3f} px)")
    print(f"Largest error range:      L{largest_range['landmark_id']} ({largest_range['error_range']:.3f} px)")
    
    # Key insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 50)
    
    # Performance consistency across pathologies
    pathology_variance = pathology_stats.groupby('landmark_id')['mean_error'].var()
    most_consistent_across_pathologies = pathology_variance.idxmin()
    most_variable_across_pathologies = pathology_variance.idxmax()
    
    print(f"• Most consistent landmark across pathologies: L{most_consistent_across_pathologies}")
    print(f"• Most variable landmark across pathologies: L{most_variable_across_pathologies}")
    
    # Overall method comparison with documented performance
    documented_error = 5.63  # From CLAUDE.md
    actual_error = overall_mean
    difference = abs(actual_error - documented_error)
    
    print(f"• Documented overall error: {documented_error:.3f} pixels")
    print(f"• Measured overall error:   {actual_error:.3f} pixels")
    print(f"• Difference: {difference:.3f} pixels ({'✅ Very close' if difference < 0.1 else '⚠️ Some variation'})")
    
    # Best performing anatomical regions
    region_performance = overall_stats.groupby('anatomical_region')['mean_error'].mean().sort_values()
    print(f"• Best performing anatomical region: {region_performance.index[0]} ({region_performance.iloc[0]:.3f} px)")
    print(f"• Worst performing anatomical region: {region_performance.index[-1]} ({region_performance.iloc[-1]:.3f} px)")
    
    print("\n📁 GENERATED FILES")
    print("-" * 50)
    files = [
        "per_landmark_statistics.csv - Detailed statistics for each landmark",
        "per_landmark_by_pathology.csv - Statistics broken down by pathology",
        "per_landmark_evaluation_report.txt - Comprehensive text report",
        "per_landmark_overall_analysis.png - Box plots and error distributions",
        "per_landmark_by_pathology.png - Pathology comparison plots",
        "error_heatmaps.png - Heatmap visualizations of errors",
        "statistical_summary.png - Statistical analysis plots",
        "best_worst_landmarks.png - Best/worst landmark analysis"
    ]
    
    for file_desc in files:
        print(f"• {file_desc}")
    
    print(f"\nAll files saved to: {evaluation_dir}")
    print("=" * 80)

if __name__ == "__main__":
    display_summary()