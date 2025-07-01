#!/usr/bin/env python3
"""
Process all 159 test images with complete ASM-style morphing using Template Matching landmarks.

This script implements the complete pipeline:
1. Load test image
2. Use Template Matching predicted landmarks (5.63px error)
3. Create Delaunay triangulation
4. Warp image to canonical/mean shape
5. Save results and visualizations
"""

import sys
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from tqdm import tqdm
import json

# Add project paths
PROJECT_ROOT = Path("/home/donrobot/Projects/Tesis")
sys.path.append(str(PROJECT_ROOT / "delaunay_morphing/src"))

from core.delaunay_lung_morpher import DelaunayLungMorpher, LungShapeMorphingAnalyzer


class ASMStyleProcessor:
    """Complete ASM-style processing using Template Matching landmarks."""
    
    def __init__(self):
        self.morpher = DelaunayLungMorpher()
        self.analyzer = LungShapeMorphingAnalyzer(self.morpher)
        self.canonical_shape = None
        self.processed_results = {}
        
    def load_template_matching_results(self):
        """Load Template Matching results with 5.63px error."""
        results_path = PROJECT_ROOT / "template_matching/results/results_coordenadas_prueba_1.pkl"
        
        with open(results_path, 'rb') as f:
            tm_results = pickle.load(f)
        
        # Load CSV for image names
        csv_path = PROJECT_ROOT / "coordenadas/coordenadas_prueba_1.csv"
        df = pd.read_csv(csv_path, header=None)
        
        # Organize data
        self.test_data = []
        predictions = tm_results['predictions']
        ground_truth = tm_results['ground_truth']
        errors = tm_results['errors']
        
        for idx in range(min(len(predictions), len(df))):
            image_name = str(df.iloc[idx, -1])
            
            self.test_data.append({
                'index': idx,
                'image_name': image_name,
                'predicted_landmarks': predictions[idx],
                'ground_truth_landmarks': ground_truth[idx],
                'error_per_landmark': errors[idx],
                'mean_error': np.mean(errors[idx])
            })
        
        print(f"Loaded {len(self.test_data)} test images")
        print(f"Overall mean error: {np.mean([d['mean_error'] for d in self.test_data]):.2f} ± {np.std([d['mean_error'] for d in self.test_data]):.2f} pixels")
        
    def compute_canonical_shape(self):
        """Compute canonical/mean shape from all Template Matching predictions."""
        print("\nComputing canonical shape from Template Matching landmarks...")
        
        all_landmarks = np.array([data['predicted_landmarks'] for data in self.test_data])
        
        # Method 1: Simple mean (like ASM mean shape)
        self.canonical_shape = np.mean(all_landmarks, axis=0)
        
        # Method 2: Procrustes-aligned mean (more robust)
        aligned_landmarks = self._procrustes_align_all(all_landmarks)
        self.canonical_shape_aligned = np.mean(aligned_landmarks, axis=0)
        
        print(f"Canonical shape computed from {len(all_landmarks)} samples")
        print(f"Shape bounds: X[{self.canonical_shape[:, 0].min():.1f}, {self.canonical_shape[:, 0].max():.1f}], Y[{self.canonical_shape[:, 1].min():.1f}, {self.canonical_shape[:, 1].max():.1f}]")
        
        return self.canonical_shape
    
    def _procrustes_align_all(self, landmarks_array):
        """Align all landmarks using Procrustes analysis."""
        reference = landmarks_array[0]  # Use first as reference
        aligned = [reference]
        
        for landmarks in landmarks_array[1:]:
            aligned_landmarks = self._procrustes_align(landmarks, reference)
            aligned.append(aligned_landmarks)
            
        return np.array(aligned)
    
    def _procrustes_align(self, landmarks, reference):
        """Align landmarks to reference using Procrustes analysis."""
        # Center both shapes
        landmarks_centered = landmarks - np.mean(landmarks, axis=0)
        reference_centered = reference - np.mean(reference, axis=0)
        
        # Scale to unit size
        landmarks_scaled = landmarks_centered / np.linalg.norm(landmarks_centered)
        reference_scaled = reference_centered / np.linalg.norm(reference_centered)
        
        # Find optimal rotation
        H = landmarks_scaled.T @ reference_scaled
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply transformation
        landmarks_rotated = landmarks_scaled @ R.T
        
        # Scale and translate to match reference
        scale = np.linalg.norm(reference_centered) / np.linalg.norm(landmarks_centered)
        landmarks_aligned = landmarks_rotated * scale + np.mean(reference, axis=0)
        
        return landmarks_aligned
    
    def load_image_by_name(self, image_name):
        """Load image from dataset."""
        for category in ['COVID', 'Normal', 'Viral Pneumonia']:
            image_path = PROJECT_ROOT / f"COVID-19_Radiography_Dataset/{category}/images/{image_name}.png"
            if image_path.exists():
                return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        return None
    
    def process_single_image(self, data_entry, save_intermediate=False):
        """Process a single image with complete ASM-style morphing."""
        image_name = data_entry['image_name']
        predicted_landmarks = data_entry['predicted_landmarks']
        
        # Load original image
        original_image = self.load_image_by_name(image_name)
        if original_image is None:
            return None
        
        try:
            # ASM-style morphing: warp image to canonical shape
            morphing_result = self.morpher.morph_image(
                original_image,
                predicted_landmarks,  # Source landmarks (Template Matching)
                self.canonical_shape,  # Target landmarks (canonical)
                output_shape=original_image.shape,
                alpha=1.0  # Complete morphing to canonical
            )
            
            warped_image = morphing_result.warped_image
            
            # Compute morphing quality metrics
            shape_diff = self.analyzer.compute_shape_difference(
                predicted_landmarks, 
                self.canonical_shape
            )
            
            triangulation_quality = self.morpher.compute_triangle_quality_metrics(
                morphing_result.triangulation
            )
            
            result = {
                'image_name': image_name,
                'original_shape': original_image.shape,
                'warped_image': warped_image,
                'original_image': original_image,
                'predicted_landmarks': predicted_landmarks,
                'canonical_landmarks': self.canonical_shape,
                'template_matching_error': data_entry['mean_error'],
                'morphing_distance': shape_diff['mean_distance'],
                'procrustes_distance': shape_diff['procrustes_mean'],
                'triangulation_quality': triangulation_quality,
                'success': True
            }
            
            # Save intermediate results if requested
            if save_intermediate:
                self._save_intermediate_visualization(result, data_entry['index'])
                
            return result
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            return {
                'image_name': image_name,
                'success': False,
                'error': str(e),
                'template_matching_error': data_entry['mean_error']
            }
    
    def _save_intermediate_visualization(self, result, index):
        """Save intermediate visualization for debugging."""
        output_dir = PROJECT_ROOT / "delaunay_morphing/processed_159"
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with landmarks
        axes[0].imshow(result['original_image'], cmap='gray')
        axes[0].scatter(result['predicted_landmarks'][:, 0], result['predicted_landmarks'][:, 1], 
                       c='red', s=20, alpha=0.7)
        axes[0].set_title(f"Original: {result['image_name']}\nTM Error: {result['template_matching_error']:.2f}px")
        axes[0].axis('off')
        
        # Warped image with canonical landmarks
        axes[1].imshow(result['warped_image'], cmap='gray')
        axes[1].scatter(result['canonical_landmarks'][:, 0], result['canonical_landmarks'][:, 1], 
                       c='blue', s=20, alpha=0.7)
        axes[1].set_title(f"Warped to Canonical\nMorph Dist: {result['morphing_distance']:.2f}px")
        axes[1].axis('off')
        
        # Triangulation overlay
        axes[2].imshow(result['original_image'], cmap='gray', alpha=0.7)
        
        # Draw triangulation
        tri = self.morpher.create_triangulation(result['predicted_landmarks'], add_boundary=False)
        for simplex in tri.simplices:
            triangle = result['predicted_landmarks'][simplex]
            triangle = np.vstack([triangle, triangle[0]])
            axes[2].plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=1, alpha=0.5)
        
        # Draw anatomical connections
        for i, j in self.morpher.contour_connections:
            axes[2].plot([result['predicted_landmarks'][i, 0], result['predicted_landmarks'][j, 0]], 
                        [result['predicted_landmarks'][i, 1], result['predicted_landmarks'][j, 1]], 
                        'g-', linewidth=2, alpha=0.8)
        
        axes[2].scatter(result['predicted_landmarks'][:, 0], result['predicted_landmarks'][:, 1], 
                       c='red', s=30)
        axes[2].set_title(f"Triangulation\nMin Angle: {result['triangulation_quality']['min_angle']:.1f}°")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"processed_{index:03d}_{result['image_name']}.png", 
                   dpi=100, bbox_inches='tight')
        plt.close()
    
    def process_all_images(self, save_every_n=20):
        """Process all 159 test images."""
        print(f"\nProcessing all {len(self.test_data)} test images...")
        print("This implements complete ASM-style morphing:")
        print("  Image → TM Landmarks → Delaunay Triangulation → Warp to Canonical")
        
        results = []
        failed_count = 0
        
        # Create output directory
        output_dir = PROJECT_ROOT / "delaunay_morphing/processed_159"
        output_dir.mkdir(exist_ok=True)
        
        for i, data_entry in enumerate(tqdm(self.test_data, desc="Processing images")):
            # Save intermediate visualizations for first few and every N images
            save_intermediate = (i < 10) or (i % save_every_n == 0)
            
            result = self.process_single_image(data_entry, save_intermediate)
            
            if result:
                results.append(result)
                if not result['success']:
                    failed_count += 1
            else:
                failed_count += 1
        
        print(f"\nProcessing completed:")
        print(f"  Successfully processed: {len(results) - failed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Success rate: {(len(results) - failed_count) / len(results) * 100:.1f}%")
        
        # Save results
        self.processed_results = results
        self._save_processing_results()
        
        return results
    
    def _save_processing_results(self):
        """Save processing results and statistics."""
        output_dir = PROJECT_ROOT / "delaunay_morphing/processed_159"
        
        # Prepare data for JSON serialization
        json_results = []
        for result in self.processed_results:
            if result['success']:
                json_result = {
                    'image_name': result['image_name'],
                    'template_matching_error': float(result['template_matching_error']),
                    'morphing_distance': float(result['morphing_distance']),
                    'procrustes_distance': float(result['procrustes_distance']),
                    'min_triangle_angle': float(result['triangulation_quality']['min_angle']),
                    'success': True
                }
            else:
                json_result = {
                    'image_name': result['image_name'],
                    'template_matching_error': float(result['template_matching_error']),
                    'success': False,
                    'error': result['error']
                }
            json_results.append(json_result)
        
        # Save JSON results
        with open(output_dir / "processing_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save canonical shape
        np.save(output_dir / "canonical_shape.npy", self.canonical_shape)
        
        print(f"Results saved to {output_dir}")
    
    def create_summary_visualization(self):
        """Create summary visualization of all processed images."""
        print("\nCreating summary visualization...")
        
        successful_results = [r for r in self.processed_results if r['success']]
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # Collect statistics
        tm_errors = [r['template_matching_error'] for r in successful_results]
        morph_distances = [r['morphing_distance'] for r in successful_results]
        triangle_angles = [r['triangulation_quality']['min_angle'] for r in successful_results]
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Template Matching Error Distribution
        plt.subplot(2, 4, 1)
        plt.hist(tm_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(np.mean(tm_errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(tm_errors):.2f}px')
        plt.xlabel('Template Matching Error (pixels)')
        plt.ylabel('Count')
        plt.title('TM Error Distribution\n(Expected: 5.63±1.03px)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Morphing Distance Distribution
        plt.subplot(2, 4, 2)
        plt.hist(morph_distances, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.mean(morph_distances), color='red', linestyle='--',
                   label=f'Mean: {np.mean(morph_distances):.1f}px')
        plt.xlabel('Morphing Distance (pixels)')
        plt.ylabel('Count')
        plt.title('Distance to Canonical Shape')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Triangle Quality
        plt.subplot(2, 4, 3)
        plt.hist(triangle_angles, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(np.mean(triangle_angles), color='red', linestyle='--',
                   label=f'Mean: {np.mean(triangle_angles):.1f}°')
        plt.xlabel('Min Triangle Angle (degrees)')
        plt.ylabel('Count')
        plt.title('Triangulation Quality')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Error vs Distance Correlation
        plt.subplot(2, 4, 4)
        plt.scatter(tm_errors, morph_distances, alpha=0.6, s=30)
        correlation = np.corrcoef(tm_errors, morph_distances)[0, 1]
        plt.xlabel('TM Error (pixels)')
        plt.ylabel('Morphing Distance (pixels)')
        plt.title(f'Error vs Distance\nCorrelation: {correlation:.3f}')
        plt.grid(True, alpha=0.3)
        
        # 5-8. Show some examples of warped images
        examples_to_show = [0, len(successful_results)//4, len(successful_results)//2, len(successful_results)*3//4]
        
        for i, idx in enumerate(examples_to_show):
            if idx < len(successful_results):
                plt.subplot(2, 4, 5 + i)
                result = successful_results[idx]
                plt.imshow(result['warped_image'], cmap='gray')
                plt.title(f"{result['image_name']}\nTM: {result['template_matching_error']:.2f}px")
                plt.axis('off')
        
        plt.suptitle(f'ASM-Style Processing Results: {len(successful_results)} Images\n' + 
                    f'Mean TM Error: {np.mean(tm_errors):.2f}±{np.std(tm_errors):.2f}px', 
                    fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        output_path = PROJECT_ROOT / "delaunay_morphing/processing_summary.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Summary visualization saved to {output_path}")
        
        # Print final statistics
        print(f"\n" + "="*60)
        print(f"FINAL PROCESSING STATISTICS")
        print(f"="*60)
        print(f"Total images processed: {len(successful_results)}")
        print(f"Template Matching error: {np.mean(tm_errors):.2f} ± {np.std(tm_errors):.2f} pixels")
        print(f"Morphing distance: {np.mean(morph_distances):.1f} ± {np.std(morph_distances):.1f} pixels")
        print(f"Triangle quality: {np.mean(triangle_angles):.1f}° ± {np.std(triangle_angles):.1f}°")
        print(f"Error-Distance correlation: {np.corrcoef(tm_errors, morph_distances)[0, 1]:.3f}")
        
        return successful_results


def main():
    """Main processing pipeline."""
    print("ASM-Style Processing of 159 Test Images")
    print("=======================================")
    print("Pipeline: Image → Template Matching Landmarks → Delaunay Morphing → Canonical Shape")
    print()
    
    # Initialize processor
    processor = ASMStyleProcessor()
    
    # Step 1: Load Template Matching results
    processor.load_template_matching_results()
    
    # Step 2: Compute canonical shape
    processor.compute_canonical_shape()
    
    # Step 3: Process all images
    results = processor.process_all_images(save_every_n=15)
    
    # Step 4: Create summary
    processor.create_summary_visualization()
    
    print("\n✓ Complete ASM-style processing finished!")
    print(f"✓ Results available in: {PROJECT_ROOT}/delaunay_morphing/processed_159/")


if __name__ == "__main__":
    main()