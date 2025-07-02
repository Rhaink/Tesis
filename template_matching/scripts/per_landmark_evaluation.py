#!/usr/bin/env python3
"""
Comprehensive per-landmark precision evaluation for Template Matching method.
Evaluates precision from every landmark compared to ground truth and provides
detailed statistics including pathology breakdown.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
import pickle
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")
SRC_DIR_TM = os.path.join(PROJECT_ROOT_DIR, "template_matching", "src")

if SRC_DIR_PULMONES not in sys.path:
    sys.path.insert(0, SRC_DIR_PULMONES)
if SRC_DIR_TM not in sys.path:
    sys.path.insert(0, SRC_DIR_TM)

from utils.evaluation import LandmarkEvaluator

class PerLandmarkEvaluator:
    """Enhanced evaluator for per-landmark precision analysis."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.project_root = Path(PROJECT_ROOT_DIR)
        self.tm_dir = self.project_root / "template_matching"
        self.results_dir = self.tm_dir / "evaluation"
        self.results_dir.mkdir(exist_ok=True)
        
        # Landmark names for interpretability
        self.landmark_names = [
            f"Landmark_{i:02d}" for i in range(15)
        ]
        
        # Anatomical region mapping
        self.landmark_regions = {
            0: "Right_Top", 1: "Left_Top", 2: "Right_Lower", 3: "Right_Middle", 
            4: "Right_Middle_Low", 5: "Right_Mid", 6: "Left_Middle_Low", 
            7: "Right_Upper", 8: "Right_Medial_Top", 9: "Center_Medial", 
            10: "Left_Medial_Top", 11: "Right_Lower_Edge", 12: "Right_Upper_Mid",
            13: "Left_Upper_Mid", 14: "Left_Upper"
        }
    
    def load_test_data(self, results_file: str = "results_coordenadas_prueba_1.pkl") -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load test dataset with ground truth coordinates from Template Matching results.
        
        Returns:
            Tuple of (ground_truth_landmarks, image_names, pathologies)
        """
        results_path = self.tm_dir / "results" / results_file
        
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract data from results
        ground_truth_landmarks = results['ground_truth']
        image_names = results['image_names']
        
        # Extract pathologies from image names
        pathologies = []
        for image_name in image_names:
            if '-' in image_name:
                # Split on last dash to handle "Viral Pneumonia-XXX" correctly
                pathology = '-'.join(image_name.split('-')[:-1])
            else:
                pathology = "Unknown"
            pathologies.append(pathology)
        
        # Convert to numpy array and ensure proper format
        gt_array = []
        for landmarks in ground_truth_landmarks:
            if isinstance(landmarks, list):
                landmarks = np.array(landmarks)
            
            # Ensure landmarks are in proper shape (15, 2)
            if landmarks.shape == (30,):
                landmarks = landmarks.reshape(15, 2)
            
            gt_array.append(landmarks)
        
        return np.array(gt_array), image_names, pathologies
    
    def load_template_matching_results(self, results_file: str = "results_coordenadas_prueba_1.pkl") -> np.ndarray:
        """
        Load Template Matching prediction results.
        
        Returns:
            Array of predicted landmarks
        """
        results_path = self.tm_dir / "results" / results_file
        
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        # Extract predictions from the results dictionary
        predictions = results['predictions']
        
        # Convert to numpy array and ensure proper format
        predictions_array = []
        for landmarks in predictions:
            if isinstance(landmarks, list):
                landmarks = np.array(landmarks)
            
            # Ensure landmarks are in proper shape (15, 2)
            if landmarks.shape == (30,):
                landmarks = landmarks.reshape(15, 2)
            
            predictions_array.append(landmarks)
        
        return np.array(predictions_array)
    
    def compute_per_landmark_errors(self, predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance error for each landmark individually.
        
        Args:
            predicted: Shape (n_samples, n_landmarks, 2)
            ground_truth: Shape (n_samples, n_landmarks, 2)
            
        Returns:
            Array of shape (n_samples, n_landmarks) with per-landmark errors
        """
        # Compute Euclidean distance for each landmark
        differences = predicted - ground_truth
        distances = np.sqrt(np.sum(differences**2, axis=2))
        return distances
    
    def analyze_per_landmark_statistics(self, 
                                      per_landmark_errors: np.ndarray,
                                      pathologies: List[str],
                                      image_names: List[str]) -> Dict:
        """
        Compute comprehensive statistics for each landmark.
        
        Returns:
            Dictionary with detailed per-landmark statistics
        """
        n_samples, n_landmarks = per_landmark_errors.shape
        
        # Overall statistics per landmark
        landmark_stats = {}
        
        for landmark_idx in range(n_landmarks):
            errors = per_landmark_errors[:, landmark_idx]
            
            stats_dict = {
                'landmark_id': landmark_idx,
                'landmark_name': self.landmark_names[landmark_idx],
                'anatomical_region': self.landmark_regions[landmark_idx],
                'n_samples': len(errors),
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'median_error': np.median(errors),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'q25_error': np.percentile(errors, 25),
                'q75_error': np.percentile(errors, 75),
                'all_errors': errors.tolist()
            }
            
            landmark_stats[landmark_idx] = stats_dict
        
        # Statistics by pathology
        pathology_stats = {}
        unique_pathologies = list(set(pathologies))
        
        for pathology in unique_pathologies:
            pathology_mask = np.array([p == pathology for p in pathologies])
            pathology_errors = per_landmark_errors[pathology_mask]
            
            pathology_landmark_stats = {}
            for landmark_idx in range(n_landmarks):
                errors = pathology_errors[:, landmark_idx]
                
                pathology_landmark_stats[landmark_idx] = {
                    'n_samples': len(errors),
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'median_error': np.median(errors),
                    'all_errors': errors.tolist()
                }
            
            pathology_stats[pathology] = pathology_landmark_stats
        
        return {
            'overall_landmark_stats': landmark_stats,
            'pathology_landmark_stats': pathology_stats,
            'unique_pathologies': unique_pathologies,
            'total_samples': n_samples,
            'total_landmarks': n_landmarks
        }
    
    def create_comprehensive_visualizations(self, 
                                          per_landmark_errors: np.ndarray,
                                          pathologies: List[str],
                                          stats_dict: Dict) -> None:
        """
        Create comprehensive visualizations for per-landmark analysis.
        """
        # Set up the plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        # 1. Overall per-landmark error distribution
        self._plot_overall_landmark_errors(per_landmark_errors, stats_dict)
        
        # 2. Per-landmark error by pathology
        self._plot_pathology_comparison(per_landmark_errors, pathologies, stats_dict)
        
        # 3. Heatmap of errors
        self._plot_error_heatmap(per_landmark_errors, pathologies, stats_dict)
        
        # 4. Statistical summary plots
        self._plot_statistical_summary(stats_dict)
        
        # 5. Best and worst performing landmarks
        self._plot_best_worst_landmarks(stats_dict)
    
    def _plot_overall_landmark_errors(self, per_landmark_errors: np.ndarray, stats_dict: Dict) -> None:
        """Plot overall per-landmark error statistics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot of errors per landmark
        landmark_errors_list = [per_landmark_errors[:, i] for i in range(15)]
        landmark_labels = [f"L{i}" for i in range(15)]
        
        box_plot = ax1.boxplot(landmark_errors_list, labels=landmark_labels, patch_artist=True)
        for patch in box_plot['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax1.set_xlabel('Landmark Index')
        ax1.set_ylabel('Error (pixels)')
        ax1.set_title('Per-Landmark Error Distribution (Box Plot)')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Mean error per landmark with error bars
        landmarks = list(range(15))
        means = [stats_dict['overall_landmark_stats'][i]['mean_error'] for i in landmarks]
        stds = [stats_dict['overall_landmark_stats'][i]['std_error'] for i in landmarks]
        
        bars = ax2.bar(landmarks, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        ax2.set_xlabel('Landmark Index')
        ax2.set_ylabel('Mean Error (pixels)')
        ax2.set_title('Mean Error per Landmark (with std dev)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Violin plot for distribution shape
        violin_parts = ax3.violinplot(landmark_errors_list, positions=landmarks, showmeans=True)
        ax3.set_xlabel('Landmark Index')
        ax3.set_ylabel('Error (pixels)')
        ax3.set_title('Per-Landmark Error Distribution (Violin Plot)')
        ax3.set_xticks(landmarks)
        ax3.set_xticklabels([f"L{i}" for i in landmarks])
        ax3.grid(True, alpha=0.3)
        
        # Cumulative error curves per landmark
        colors = plt.cm.tab20(np.linspace(0, 1, 15))
        for i, color in enumerate(colors):
            errors = per_landmark_errors[:, i]
            sorted_errors = np.sort(errors)
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax4.plot(sorted_errors, cumulative, label=f'L{i}', color=color, alpha=0.7)
        
        ax4.set_xlabel('Error Threshold (pixels)')
        ax4.set_ylabel('Cumulative Proportion')
        ax4.set_title('Cumulative Error Distribution per Landmark')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "per_landmark_overall_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pathology_comparison(self, per_landmark_errors: np.ndarray, 
                                 pathologies: List[str], stats_dict: Dict) -> None:
        """Plot per-landmark errors by pathology."""
        unique_pathologies = stats_dict['unique_pathologies']
        n_pathologies = len(unique_pathologies)
        
        fig, axes = plt.subplots(n_pathologies, 1, figsize=(16, 5 * n_pathologies))
        if n_pathologies == 1:
            axes = [axes]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        for i, pathology in enumerate(unique_pathologies):
            pathology_mask = np.array([p == pathology for p in pathologies])
            pathology_errors = per_landmark_errors[pathology_mask]
            
            # Box plot for this pathology
            landmark_errors_list = [pathology_errors[:, j] for j in range(15)]
            landmark_labels = [f"L{j}" for j in range(15)]
            
            box_plot = axes[i].boxplot(landmark_errors_list, labels=landmark_labels, 
                                     patch_artist=True)
            for patch in box_plot['boxes']:
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
            
            axes[i].set_xlabel('Landmark Index')
            axes[i].set_ylabel('Error (pixels)')
            axes[i].set_title(f'Per-Landmark Errors - {pathology} (n={np.sum(pathology_mask)})')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "per_landmark_by_pathology.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_heatmap(self, per_landmark_errors: np.ndarray, 
                          pathologies: List[str], stats_dict: Dict) -> None:
        """Create heatmap visualization of errors."""
        # Create heatmap of mean errors by pathology and landmark
        unique_pathologies = stats_dict['unique_pathologies']
        heatmap_data = np.zeros((len(unique_pathologies), 15))
        
        for i, pathology in enumerate(unique_pathologies):
            pathology_mask = np.array([p == pathology for p in pathologies])
            pathology_errors = per_landmark_errors[pathology_mask]
            heatmap_data[i, :] = np.mean(pathology_errors, axis=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # Heatmap 1: Mean errors by pathology and landmark
        im1 = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(15))
        ax1.set_xticklabels([f"L{i}" for i in range(15)])
        ax1.set_yticks(range(len(unique_pathologies)))
        ax1.set_yticklabels(unique_pathologies)
        ax1.set_xlabel('Landmark Index')
        ax1.set_ylabel('Pathology')
        ax1.set_title('Mean Error Heatmap by Pathology and Landmark')
        
        # Add text annotations
        for i in range(len(unique_pathologies)):
            for j in range(15):
                text = ax1.text(j, i, f'{heatmap_data[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im1, ax=ax1, label='Mean Error (pixels)')
        
        # Heatmap 2: Overall landmark error ranking
        overall_means = [stats_dict['overall_landmark_stats'][i]['mean_error'] for i in range(15)]
        landmark_ranking = np.argsort(overall_means)
        
        ranking_data = np.array(overall_means)[landmark_ranking].reshape(1, -1)
        im2 = ax2.imshow(ranking_data, cmap='RdYlGn_r', aspect='auto')
        ax2.set_xticks(range(15))
        ax2.set_xticklabels([f"L{landmark_ranking[i]}" for i in range(15)])
        ax2.set_yticks([0])
        ax2.set_yticklabels(['All Pathologies'])
        ax2.set_xlabel('Landmark Index (sorted by performance)')
        ax2.set_title('Landmark Performance Ranking (Best to Worst)')
        
        # Add text annotations
        for j in range(15):
            text = ax2.text(j, 0, f'{ranking_data[0, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im2, ax=ax2, label='Mean Error (pixels)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "error_heatmaps.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_summary(self, stats_dict: Dict) -> None:
        """Plot statistical summary of landmarks."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        landmarks = list(range(15))
        
        # Statistics by metric
        means = [stats_dict['overall_landmark_stats'][i]['mean_error'] for i in landmarks]
        medians = [stats_dict['overall_landmark_stats'][i]['median_error'] for i in landmarks]
        stds = [stats_dict['overall_landmark_stats'][i]['std_error'] for i in landmarks]
        maxs = [stats_dict['overall_landmark_stats'][i]['max_error'] for i in landmarks]
        
        # Mean vs Median
        ax1.scatter(means, medians, s=100, alpha=0.7, c='blue')
        ax1.plot([min(means), max(means)], [min(means), max(means)], 'r--', alpha=0.5)
        ax1.set_xlabel('Mean Error (pixels)')
        ax1.set_ylabel('Median Error (pixels)')
        ax1.set_title('Mean vs Median Error per Landmark')
        ax1.grid(True, alpha=0.3)
        
        for i, (mean, median) in enumerate(zip(means, medians)):
            ax1.annotate(f'L{i}', (mean, median), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # Standard Deviation
        ax2.bar(landmarks, stds, alpha=0.7, color='orange')
        ax2.set_xlabel('Landmark Index')
        ax2.set_ylabel('Standard Deviation (pixels)')
        ax2.set_title('Error Variability per Landmark')
        ax2.grid(True, alpha=0.3)
        
        # Error Range (Min to Max)
        mins = [stats_dict['overall_landmark_stats'][i]['min_error'] for i in landmarks]
        error_ranges = [(maxs[i] - mins[i]) for i in landmarks]
        
        ax3.bar(landmarks, error_ranges, alpha=0.7, color='red')
        ax3.set_xlabel('Landmark Index')
        ax3.set_ylabel('Error Range (pixels)')
        ax3.set_title('Error Range (Max - Min) per Landmark')
        ax3.grid(True, alpha=0.3)
        
        # Coefficient of Variation
        cvs = [stds[i] / means[i] if means[i] > 0 else 0 for i in landmarks]
        ax4.bar(landmarks, cvs, alpha=0.7, color='purple')
        ax4.set_xlabel('Landmark Index')
        ax4.set_ylabel('Coefficient of Variation')
        ax4.set_title('Relative Variability per Landmark')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "statistical_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_best_worst_landmarks(self, stats_dict: Dict) -> None:
        """Plot analysis of best and worst performing landmarks."""
        landmarks = list(range(15))
        means = [stats_dict['overall_landmark_stats'][i]['mean_error'] for i in landmarks]
        
        # Sort landmarks by performance
        sorted_indices = np.argsort(means)
        best_3 = sorted_indices[:3]
        worst_3 = sorted_indices[-3:]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Best performing landmarks
        best_errors = [stats_dict['overall_landmark_stats'][i]['all_errors'] for i in best_3]
        best_labels = [f"L{i} ({means[i]:.2f}px)" for i in best_3]
        
        violin_parts = ax1.violinplot(best_errors, positions=range(3), showmeans=True)
        ax1.set_xticks(range(3))
        ax1.set_xticklabels(best_labels)
        ax1.set_ylabel('Error (pixels)')
        ax1.set_title('Best Performing Landmarks (Lowest Mean Error)')
        ax1.grid(True, alpha=0.3)
        
        # Worst performing landmarks
        worst_errors = [stats_dict['overall_landmark_stats'][i]['all_errors'] for i in worst_3]
        worst_labels = [f"L{i} ({means[i]:.2f}px)" for i in worst_3]
        
        violin_parts = ax2.violinplot(worst_errors, positions=range(3), showmeans=True)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(worst_labels)
        ax2.set_ylabel('Error (pixels)')
        ax2.set_title('Worst Performing Landmarks (Highest Mean Error)')
        ax2.grid(True, alpha=0.3)
        
        # Performance ranking
        ax3.barh(range(15), [means[i] for i in sorted_indices], 
                color=['green' if i < 5 else 'orange' if i < 10 else 'red' for i in range(15)])
        ax3.set_yticks(range(15))
        ax3.set_yticklabels([f"L{sorted_indices[i]}" for i in range(15)])
        ax3.set_xlabel('Mean Error (pixels)')
        ax3.set_title('Landmark Performance Ranking')
        ax3.grid(True, alpha=0.3)
        
        # Add anatomical region information
        anatomical_means = {}
        for i in landmarks:
            region = stats_dict['overall_landmark_stats'][i]['anatomical_region']
            if region not in anatomical_means:
                anatomical_means[region] = []
            anatomical_means[region].append(means[i])
        
        region_means = {region: np.mean(errors) for region, errors in anatomical_means.items()}
        
        regions = list(region_means.keys())
        region_values = list(region_means.values())
        
        ax4.bar(range(len(regions)), region_values, alpha=0.7)
        ax4.set_xticks(range(len(regions)))
        ax4.set_xticklabels(regions, rotation=45, ha='right')
        ax4.set_ylabel('Mean Error (pixels)')
        ax4.set_title('Error by Anatomical Region')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "best_worst_landmarks.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self, stats_dict: Dict, pathologies: List[str]) -> str:
        """Generate a comprehensive text report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TEMPLATE MATCHING - PER-LANDMARK PRECISION EVALUATION")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall summary
        overall_mean = np.mean([stats_dict['overall_landmark_stats'][i]['mean_error'] 
                               for i in range(15)])
        overall_std = np.std([stats_dict['overall_landmark_stats'][i]['mean_error'] 
                             for i in range(15)])
        
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total samples: {stats_dict['total_samples']}")
        report_lines.append(f"Total landmarks: {stats_dict['total_landmarks']}")
        report_lines.append(f"Average error across all landmarks: {overall_mean:.3f} ± {overall_std:.3f} pixels")
        report_lines.append("")
        
        # Per-landmark detailed statistics
        report_lines.append("PER-LANDMARK DETAILED STATISTICS")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Landmark':<10} {'Region':<20} {'Mean':<8} {'Std':<8} {'Median':<8} {'Min':<8} {'Max':<8}")
        report_lines.append("-" * 80)
        
        for i in range(15):
            stats = stats_dict['overall_landmark_stats'][i]
            report_lines.append(
                f"L{i:<9} {stats['anatomical_region']:<20} "
                f"{stats['mean_error']:<8.3f} {stats['std_error']:<8.3f} "
                f"{stats['median_error']:<8.3f} {stats['min_error']:<8.3f} {stats['max_error']:<8.3f}"
            )
        report_lines.append("")
        
        # Best and worst landmarks
        landmarks = list(range(15))
        means = [stats_dict['overall_landmark_stats'][i]['mean_error'] for i in landmarks]
        sorted_indices = np.argsort(means)
        
        report_lines.append("LANDMARK PERFORMANCE RANKING")
        report_lines.append("-" * 40)
        report_lines.append("Best performing landmarks (lowest error):")
        for i in range(3):
            idx = sorted_indices[i]
            report_lines.append(f"  {i+1}. Landmark {idx}: {means[idx]:.3f} pixels")
        
        report_lines.append("")
        report_lines.append("Worst performing landmarks (highest error):")
        for i in range(3):
            idx = sorted_indices[-(i+1)]
            report_lines.append(f"  {i+1}. Landmark {idx}: {means[idx]:.3f} pixels")
        report_lines.append("")
        
        # Pathology breakdown
        report_lines.append("PATHOLOGY BREAKDOWN")
        report_lines.append("-" * 40)
        
        for pathology in stats_dict['unique_pathologies']:
            pathology_stats = stats_dict['pathology_landmark_stats'][pathology]
            pathology_count = len([p for p in pathologies if p == pathology])
            
            # Calculate overall statistics for this pathology
            pathology_means = [pathology_stats[i]['mean_error'] for i in range(15)]
            pathology_overall_mean = np.mean(pathology_means)
            pathology_overall_std = np.std(pathology_means)
            
            report_lines.append(f"{pathology} (n={pathology_count}):")
            report_lines.append(f"  Overall mean: {pathology_overall_mean:.3f} ± {pathology_overall_std:.3f} pixels")
            
            # Best and worst landmarks for this pathology
            pathology_sorted = np.argsort(pathology_means)
            report_lines.append(f"  Best landmark: L{pathology_sorted[0]} ({pathology_means[pathology_sorted[0]]:.3f} px)")
            report_lines.append(f"  Worst landmark: L{pathology_sorted[-1]} ({pathology_means[pathology_sorted[-1]]:.3f} px)")
            report_lines.append("")
        
        # Statistical significance tests between pathologies
        if len(stats_dict['unique_pathologies']) > 1:
            report_lines.append("STATISTICAL ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append("Pairwise comparisons between pathologies (Wilcoxon rank-sum test):")
            
            pathologies_array = np.array(pathologies)
            unique_pathologies = stats_dict['unique_pathologies']
            
            for i, path1 in enumerate(unique_pathologies):
                for j, path2 in enumerate(unique_pathologies[i+1:], i+1):
                    mask1 = pathologies_array == path1
                    mask2 = pathologies_array == path2
                    
                    # Get all errors for each pathology
                    all_errors1 = []
                    all_errors2 = []
                    
                    for landmark_idx in range(15):
                        all_errors1.extend(stats_dict['pathology_landmark_stats'][path1][landmark_idx]['all_errors'])
                        all_errors2.extend(stats_dict['pathology_landmark_stats'][path2][landmark_idx]['all_errors'])
                    
                    try:
                        statistic, p_value = stats.mannwhitneyu(all_errors1, all_errors2, alternative='two-sided')
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        report_lines.append(f"  {path1} vs {path2}: p = {p_value:.4f} {significance}")
                    except:
                        report_lines.append(f"  {path1} vs {path2}: Could not compute")
            
            report_lines.append("")
        
        # Technical details
        report_lines.append("TECHNICAL DETAILS")
        report_lines.append("-" * 40)
        report_lines.append("• Error metric: Euclidean distance between predicted and ground truth landmarks")
        report_lines.append("• Coordinate system: 64x64 reference system")
        report_lines.append("• Template Matching method: PCA-based eigenpatches with multi-scale prediction")
        report_lines.append("• Evaluation dataset: coordenadas_prueba_1.csv (159 test images)")
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_statistics_to_csv(self, stats_dict: Dict, pathologies: List[str]) -> None:
        """Save detailed statistics to CSV files."""
        
        # 1. Overall landmark statistics
        overall_data = []
        for i in range(15):
            stats = stats_dict['overall_landmark_stats'][i]
            overall_data.append({
                'landmark_id': i,
                'landmark_name': stats['landmark_name'],
                'anatomical_region': stats['anatomical_region'],
                'n_samples': stats['n_samples'],
                'mean_error': stats['mean_error'],
                'std_error': stats['std_error'],
                'median_error': stats['median_error'],
                'min_error': stats['min_error'],
                'max_error': stats['max_error'],
                'q25_error': stats['q25_error'],
                'q75_error': stats['q75_error']
            })
        
        df_overall = pd.DataFrame(overall_data)
        df_overall.to_csv(self.results_dir / "per_landmark_statistics.csv", index=False)
        
        # 2. Pathology-specific statistics
        pathology_data = []
        for pathology in stats_dict['unique_pathologies']:
            for landmark_idx in range(15):
                pathology_stats = stats_dict['pathology_landmark_stats'][pathology][landmark_idx]
                pathology_data.append({
                    'pathology': pathology,
                    'landmark_id': landmark_idx,
                    'n_samples': pathology_stats['n_samples'],
                    'mean_error': pathology_stats['mean_error'],
                    'std_error': pathology_stats['std_error'],
                    'median_error': pathology_stats['median_error']
                })
        
        df_pathology = pd.DataFrame(pathology_data)
        df_pathology.to_csv(self.results_dir / "per_landmark_by_pathology.csv", index=False)
        
        print(f"Statistics saved to:")
        print(f"  - {self.results_dir / 'per_landmark_statistics.csv'}")
        print(f"  - {self.results_dir / 'per_landmark_by_pathology.csv'}")
    
    def run_complete_evaluation(self) -> None:
        """Run the complete per-landmark evaluation pipeline."""
        print("Starting comprehensive per-landmark evaluation for Template Matching...")
        print("=" * 80)
        
        # Load data
        print("Loading test data...")
        ground_truth, image_names, pathologies = self.load_test_data()
        print(f"Loaded {len(ground_truth)} samples with pathologies: {set(pathologies)}")
        
        print("Loading Template Matching results...")
        predictions = self.load_template_matching_results()
        print(f"Loaded {len(predictions)} predictions")
        
        # Compute per-landmark errors
        print("Computing per-landmark errors...")
        per_landmark_errors = self.compute_per_landmark_errors(predictions, ground_truth)
        print(f"Computed errors for {per_landmark_errors.shape[0]} samples and {per_landmark_errors.shape[1]} landmarks")
        
        # Analyze statistics
        print("Analyzing statistics...")
        stats_dict = self.analyze_per_landmark_statistics(per_landmark_errors, pathologies, image_names)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_comprehensive_visualizations(per_landmark_errors, pathologies, stats_dict)
        
        # Generate report
        print("Generating detailed report...")
        report = self.generate_detailed_report(stats_dict, pathologies)
        
        # Save report
        with open(self.results_dir / "per_landmark_evaluation_report.txt", 'w') as f:
            f.write(report)
        
        # Save statistics to CSV
        print("Saving statistics to CSV...")
        self.save_statistics_to_csv(stats_dict, pathologies)
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {self.results_dir}")
        print("\nGenerated files:")
        print("• per_landmark_evaluation_report.txt - Comprehensive text report")
        print("• per_landmark_statistics.csv - Overall statistics per landmark")
        print("• per_landmark_by_pathology.csv - Statistics by pathology")
        print("• per_landmark_overall_analysis.png - Overall analysis plots")
        print("• per_landmark_by_pathology.png - Pathology comparison plots")
        print("• error_heatmaps.png - Error heatmap visualizations")
        print("• statistical_summary.png - Statistical summary plots")
        print("• best_worst_landmarks.png - Best/worst landmark analysis")
        print("\n" + "=" * 80)
        
        # Print key findings
        overall_mean = np.mean([stats_dict['overall_landmark_stats'][i]['mean_error'] 
                               for i in range(15)])
        print("KEY FINDINGS:")
        print(f"• Average error across all landmarks: {overall_mean:.3f} pixels")
        
        # Best and worst landmarks
        landmarks = list(range(15))
        means = [stats_dict['overall_landmark_stats'][i]['mean_error'] for i in landmarks]
        sorted_indices = np.argsort(means)
        
        print(f"• Best performing landmark: L{sorted_indices[0]} ({means[sorted_indices[0]]:.3f} px)")
        print(f"• Worst performing landmark: L{sorted_indices[-1]} ({means[sorted_indices[-1]]:.3f} px)")
        
        # Pathology comparison
        for pathology in stats_dict['unique_pathologies']:
            pathology_stats = stats_dict['pathology_landmark_stats'][pathology]
            pathology_means = [pathology_stats[i]['mean_error'] for i in range(15)]
            pathology_overall_mean = np.mean(pathology_means)
            pathology_count = len([p for p in pathologies if p == pathology])
            print(f"• {pathology} (n={pathology_count}): {pathology_overall_mean:.3f} pixels")
        
        print("=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Per-landmark evaluation for Template Matching')
    parser.add_argument('--coords_file', default='coordenadas_prueba_1.csv',
                       help='Coordinates CSV file to use')
    parser.add_argument('--results_file', default='results_coordenadas_prueba_1.pkl',
                       help='Template Matching results pickle file')
    
    args = parser.parse_args()
    
    evaluator = PerLandmarkEvaluator()
    evaluator.run_complete_evaluation()


if __name__ == "__main__":
    main()