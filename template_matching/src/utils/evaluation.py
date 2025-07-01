"""
Evaluation utilities for comparing template matching with ASM methods.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path
import logging


class LandmarkEvaluator:
    """
    Evaluator for landmark detection methods.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def compute_point_to_point_error(self, 
                                   predicted: np.ndarray, 
                                   ground_truth: np.ndarray) -> float:
        """
        Compute point-to-point error between predicted and ground truth landmarks.
        
        Args:
            predicted: Predicted landmarks of shape (n_landmarks, 2)
            ground_truth: Ground truth landmarks of shape (n_landmarks, 2)
            
        Returns:
            Mean Euclidean distance between corresponding points
        """
        if predicted.shape != ground_truth.shape:
            raise ValueError("Predicted and ground truth shapes must match")
        
        distances = np.linalg.norm(predicted - ground_truth, axis=1)
        return np.mean(distances)
    
    def compute_cumulative_error_distribution(self, 
                                            errors: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cumulative error distribution for CED curve.
        
        Args:
            errors: List of point-to-point errors
            
        Returns:
            Tuple of (thresholds, cumulative_proportions)
        """
        errors = np.array(errors)
        thresholds = np.linspace(0, np.max(errors), 100)
        cumulative_proportions = []
        
        for thresh in thresholds:
            proportion = np.mean(errors <= thresh)
            cumulative_proportions.append(proportion)
        
        return thresholds, np.array(cumulative_proportions)
    
    def evaluate_method(self, 
                       method_name: str,
                       predicted_landmarks: List[np.ndarray],
                       ground_truth_landmarks: List[np.ndarray],
                       image_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a landmark detection method.
        
        Args:
            method_name: Name of the method being evaluated
            predicted_landmarks: List of predicted landmark arrays
            ground_truth_landmarks: List of ground truth landmark arrays
            image_names: Optional list of image names for detailed results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if len(predicted_landmarks) != len(ground_truth_landmarks):
            raise ValueError("Number of predicted and ground truth samples must match")
        
        errors = []
        detailed_results = []
        
        for i, (pred, gt) in enumerate(zip(predicted_landmarks, ground_truth_landmarks)):
            try:
                error = self.compute_point_to_point_error(pred, gt)
                errors.append(error)
                
                result = {
                    'sample_id': i,
                    'error': error,
                }
                
                if image_names:
                    result['image_name'] = image_names[i]
                
                detailed_results.append(result)
                
            except Exception as e:
                logging.warning(f"Error evaluating sample {i}: {str(e)}")
                continue
        
        if not errors:
            return {
                'method_name': method_name,
                'n_samples': 0,
                'mean_error': float('inf'),
                'std_error': 0,
                'median_error': float('inf'),
                'min_error': float('inf'),
                'max_error': float('inf'),
                'detailed_results': []
            }
        
        # Compute statistics
        errors = np.array(errors)
        
        results = {
            'method_name': method_name,
            'n_samples': len(errors),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'detailed_results': detailed_results,
            'all_errors': errors.tolist()
        }
        
        # Store results
        self.results[method_name] = results
        
        return results
    
    def compare_methods(self, method_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple methods and create summary table.
        
        Args:
            method_results: Dictionary mapping method names to evaluation results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for method_name, results in method_results.items():
            comparison_data.append({
                'Method': method_name,
                'N Samples': results['n_samples'],
                'Mean Error (px)': results['mean_error'],
                'Std Error (px)': results['std_error'],
                'Median Error (px)': results['median_error'],
                'Min Error (px)': results['min_error'],
                'Max Error (px)': results['max_error']
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_error_distributions(self, 
                               method_results: Dict[str, Dict[str, Any]],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot error distributions for multiple methods.
        
        Args:
            method_results: Dictionary mapping method names to evaluation results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        data_for_box = []
        labels_for_box = []
        
        for method_name, results in method_results.items():
            if 'all_errors' in results and results['all_errors']:
                data_for_box.append(results['all_errors'])
                labels_for_box.append(method_name)
        
        ax1.boxplot(data_for_box, labels=labels_for_box)
        ax1.set_ylabel('Point-to-Point Error (pixels)')
        ax1.set_title('Error Distribution Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative Error Distribution (CED) curves
        colors = plt.cm.Set1(np.linspace(0, 1, len(method_results)))
        
        for (method_name, results), color in zip(method_results.items(), colors):
            if 'all_errors' in results and results['all_errors']:
                thresholds, cum_props = self.compute_cumulative_error_distribution(results['all_errors'])
                ax2.plot(thresholds, cum_props, label=method_name, color=color, linewidth=2)
        
        ax2.set_xlabel('Error Threshold (pixels)')
        ax2.set_ylabel('Proportion of Samples')
        ax2.set_title('Cumulative Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(20, max([max(r['all_errors']) for r in method_results.values() if r['all_errors']])))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per_landmark_errors(self,
                               predicted_landmarks: List[np.ndarray],
                               ground_truth_landmarks: List[np.ndarray],
                               method_name: str,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot error statistics per landmark point.
        
        Args:
            predicted_landmarks: List of predicted landmark arrays
            ground_truth_landmarks: List of ground truth landmark arrays
            method_name: Name of the method
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not predicted_landmarks:
            raise ValueError("No landmarks provided")
        
        n_landmarks = predicted_landmarks[0].shape[0]
        per_landmark_errors = [[] for _ in range(n_landmarks)]
        
        # Collect errors per landmark
        for pred, gt in zip(predicted_landmarks, ground_truth_landmarks):
            distances = np.linalg.norm(pred - gt, axis=1)
            for i, dist in enumerate(distances):
                per_landmark_errors[i].append(dist)
        
        # Compute statistics
        landmark_stats = []
        for i, errors in enumerate(per_landmark_errors):
            if errors:
                landmark_stats.append({
                    'landmark': i,
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'median': np.median(errors),
                    'errors': errors
                })
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean error per landmark
        landmarks = [s['landmark'] for s in landmark_stats]
        means = [s['mean'] for s in landmark_stats]
        stds = [s['std'] for s in landmark_stats]
        
        ax1.bar(landmarks, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('Landmark Index')
        ax1.set_ylabel('Mean Error (pixels)')
        ax1.set_title(f'Per-Landmark Error Statistics - {method_name}')
        ax1.grid(True, alpha=0.3)
        
        # Box plot of errors per landmark
        error_data = [s['errors'] for s in landmark_stats]
        ax2.boxplot(error_data, labels=landmarks)
        ax2.set_xlabel('Landmark Index')
        ax2.set_ylabel('Error (pixels)')
        ax2.set_title(f'Per-Landmark Error Distribution - {method_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_detailed_results(self, 
                            method_results: Dict[str, Dict[str, Any]],
                            output_path: str):
        """
        Save detailed evaluation results to CSV.
        
        Args:
            method_results: Dictionary mapping method names to evaluation results
            output_path: Path to save the CSV file
        """
        all_detailed_results = []
        
        for method_name, results in method_results.items():
            for detail in results.get('detailed_results', []):
                detail_copy = detail.copy()
                detail_copy['method'] = method_name
                all_detailed_results.append(detail_copy)
        
        if all_detailed_results:
            df = pd.DataFrame(all_detailed_results)
            df.to_csv(output_path, index=False)
            logging.info(f"Detailed results saved to {output_path}")
        else:
            logging.warning("No detailed results to save")


class MethodComparator:
    """
    Utility class for comparing different landmark detection methods.
    """
    
    def __init__(self):
        """Initialize comparator."""
        self.evaluator = LandmarkEvaluator()
    
    def compare_template_matching_vs_asm(self,
                                       template_predictions: List[np.ndarray],
                                       asm_predictions: List[np.ndarray],
                                       ground_truth: List[np.ndarray],
                                       image_names: Optional[List[str]] = None,
                                       output_dir: str = ".") -> Dict[str, Any]:
        """
        Compare template matching method against ASM.
        
        Args:
            template_predictions: Template matching predictions
            asm_predictions: ASM predictions
            ground_truth: Ground truth landmarks
            image_names: Optional image names
            output_dir: Output directory for results
            
        Returns:
            Dictionary with comparison results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Evaluate both methods
        template_results = self.evaluator.evaluate_method(
            "Template Matching", template_predictions, ground_truth, image_names
        )
        
        asm_results = self.evaluator.evaluate_method(
            "ASM", asm_predictions, ground_truth, image_names
        )
        
        method_results = {
            "Template Matching": template_results,
            "ASM": asm_results
        }
        
        # Create comparison table
        comparison_df = self.evaluator.compare_methods(method_results)
        comparison_df.to_csv(output_dir / "method_comparison.csv", index=False)
        
        # Plot comparisons
        fig1 = self.evaluator.plot_error_distributions(
            method_results, str(output_dir / "error_distributions.png")
        )
        
        # Plot per-landmark analysis
        fig2 = self.evaluator.plot_per_landmark_errors(
            template_predictions, ground_truth, "Template Matching",
            str(output_dir / "template_per_landmark.png")
        )
        
        fig3 = self.evaluator.plot_per_landmark_errors(
            asm_predictions, ground_truth, "ASM",
            str(output_dir / "asm_per_landmark.png")
        )
        
        # Save detailed results
        self.evaluator.save_detailed_results(
            method_results, str(output_dir / "detailed_results.csv")
        )
        
        # Statistical significance test
        from scipy.stats import wilcoxon
        template_errors = template_results['all_errors']
        asm_errors = asm_results['all_errors']
        
        if len(template_errors) == len(asm_errors) and len(template_errors) > 0:
            try:
                stat, p_value = wilcoxon(template_errors, asm_errors)
                significance_test = {
                    'test': 'Wilcoxon signed-rank test',
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except:
                significance_test = {'test': 'Failed', 'error': 'Could not perform test'}
        else:
            significance_test = {'test': 'Not applicable', 'reason': 'Different sample sizes'}
        
        return {
            'template_results': template_results,
            'asm_results': asm_results,
            'comparison_table': comparison_df,
            'significance_test': significance_test,
            'output_files': {
                'comparison_csv': str(output_dir / "method_comparison.csv"),
                'detailed_csv': str(output_dir / "detailed_results.csv"),
                'error_distributions': str(output_dir / "error_distributions.png"),
                'template_per_landmark': str(output_dir / "template_per_landmark.png"),
                'asm_per_landmark': str(output_dir / "asm_per_landmark.png")
            }
        }