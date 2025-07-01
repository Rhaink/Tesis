#!/usr/bin/env python3
"""
Fixed Interactive viewer that uses the ACTUAL saved results with 5.63px error.
"""

import sys
import os
import numpy as np
import cv2
import pickle
import logging
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import argparse

# Setup paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")
sys.path.insert(0, SRC_DIR_PULMONES)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TemplateLandmarkViewer:
    """Interactive viewer for template matching results using REAL saved predictions."""
    
    def __init__(self):
        self.images = []
        self.ground_truth = []
        self.predictions = []
        self.image_names = []
        self.current_idx = 0
        self.fig = None
        self.ax = None
        
    def load_saved_results(self, num_samples=10):
        """Load saved results from process_all_images.py output."""
        from utils import asm_utils
        
        # Load the saved results file
        results_file = os.path.join(PROJECT_ROOT_DIR, 'template_matching', 'results', 
                                   'results_coordenadas_prueba_1.pkl')
        
        if not os.path.exists(results_file):
            logging.error(f"Results file not found: {results_file}")
            logging.info("Run process_all_images.py first to generate results")
            return False
            
        # Load results
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
            
        logging.info(f"Loaded results for {len(results['predictions'])} images")
        
        # Take only requested samples
        if num_samples and num_samples < len(results['predictions']):
            indices = np.random.choice(len(results['predictions']), num_samples, replace=False)
            indices = sorted(indices)  # Keep order
        else:
            indices = list(range(len(results['predictions'])))
            
        # Load images for selected indices
        images_base_dir = os.path.join(PROJECT_ROOT_DIR, 'COVID-19_Radiography_Dataset')
        
        for idx in indices:
            img_name = results['image_names'][idx]
            pred_landmarks = results['predictions'][idx]
            gt_landmarks = results['ground_truth'][idx]
            
            # Load actual image
            img_path = asm_utils.get_image_path(img_name, None, images_base_dir)
            if img_path and os.path.exists(img_path):
                image = asm_utils.load_image_grayscale(img_path)
                if image is not None:
                    self.images.append(image)
                    self.predictions.append(pred_landmarks)
                    self.ground_truth.append(gt_landmarks)
                    self.image_names.append(img_name)
        
        logging.info(f"Successfully loaded {len(self.images)} images for viewing")
        return True
    
    def plot_current_image(self):
        """Plot current image with landmarks."""
        if not self.images:
            return
        
        self.ax.clear()
        
        # Get current data
        image = self.images[self.current_idx]
        true_lm = self.ground_truth[self.current_idx]
        pred_lm = self.predictions[self.current_idx]
        img_name = self.image_names[self.current_idx]
        
        # Display image
        self.ax.imshow(image, cmap='gray')
        
        # Plot landmarks - SAME STYLE AS BEFORE
        self.ax.scatter(true_lm[:, 0], true_lm[:, 1], c='lime', s=150, alpha=0.9,
                       label='Ground Truth', marker='o', edgecolors='darkgreen', linewidth=3)
        
        self.ax.scatter(pred_lm[:, 0], pred_lm[:, 1], c='red', s=150, alpha=0.9,
                       label='Predicted', marker='x', linewidth=4)
        
        # Draw error lines
        for j, (true_pt, pred_pt) in enumerate(zip(true_lm, pred_lm)):
            self.ax.plot([true_pt[0], pred_pt[0]], [true_pt[1], pred_pt[1]], 
                        'yellow', alpha=0.6, linewidth=2)
            
            # Add landmark numbers
            self.ax.annotate(f'{j}', (true_pt[0], true_pt[1]), xytext=(8, 8), 
                           textcoords='offset points', fontsize=12, color='white', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.8))
        
        # Compute error (should match saved results)
        error = np.mean(np.linalg.norm(pred_lm - true_lm, axis=1))
        max_error = np.max(np.linalg.norm(pred_lm - true_lm, axis=1))
        
        # Set title with detailed info
        self.ax.set_title(f'Template Matching Results [{self.current_idx+1}/{len(self.images)}]\n'
                         f'{img_name}\n'
                         f'Mean Error: {error:.2f} px | Max Error: {max_error:.2f} px',
                         fontsize=11, fontweight='bold')
        
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('X coordinate (pixels)')
        self.ax.set_ylabel('Y coordinate (pixels)')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def next_image(self, event):
        """Go to next image."""
        self.current_idx = (self.current_idx + 1) % len(self.images)
        self.plot_current_image()
    
    def prev_image(self, event):
        """Go to previous image."""
        self.current_idx = (self.current_idx - 1) % len(self.images)
        self.plot_current_image()
    
    def show_stats(self, event):
        """Show detailed statistics."""
        if not self.predictions:
            return
        
        errors = []
        for true_lm, pred_lm in zip(self.ground_truth, self.predictions):
            error = np.mean(np.linalg.norm(pred_lm - true_lm, axis=1))
            errors.append(error)
        
        # Create stats window
        stats_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error per image
        ax1.bar(range(len(errors)), errors, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_title('Error per Image')
        ax1.set_xlabel('Image Index')
        ax1.set_ylabel('Mean Error (pixels)')
        ax1.grid(True, alpha=0.3)
        
        # Highlight current image
        ax1.bar(self.current_idx, errors[self.current_idx], alpha=1.0, color='red', edgecolor='darkred')
        
        # Add mean line
        mean_error = np.mean(errors)
        ax1.axhline(y=mean_error, color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_error:.2f} px')
        ax1.legend()
        
        # Error distribution
        ax2.hist(errors, bins=10, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax2.axvline(mean_error, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_error:.2f}')
        ax2.set_title('Error Distribution')
        ax2.set_xlabel('Mean Error (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Template Matching Performance Statistics\n'
                    f'Showing {len(errors)} images from test set (159 total)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def run_interactive_viewer(self, num_samples=10):
        """Run the interactive viewer with REAL results."""
        print("🔄 Loading REAL test results (not simulation)...")
        
        success = self.load_saved_results(num_samples)
        if not success or not self.images:
            print("❌ Could not load test results")
            return
        
        print(f"✅ Loaded {len(self.images)} images with actual predictions")
        print(f"📊 These results have ~5.63 px average error")
        
        # Create main figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.15)
        
        # Add control buttons
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.25, 0.05, 0.1, 0.04])
        ax_stats = plt.axes([0.4, 0.05, 0.15, 0.04])
        ax_quit = plt.axes([0.8, 0.05, 0.1, 0.04])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        btn_stats = Button(ax_stats, 'Statistics')
        btn_quit = Button(ax_quit, 'Quit')
        
        btn_prev.on_clicked(self.prev_image)
        btn_next.on_clicked(self.next_image)
        btn_stats.on_clicked(self.show_stats)
        btn_quit.on_clicked(lambda x: plt.close('all'))
        
        # Initial plot
        self.plot_current_image()
        
        print("🎉 Interactive viewer ready with REAL results!")
        print("📖 Instructions:")
        print("   • Use 'Previous'/'Next' buttons to navigate")
        print("   • Click 'Statistics' to see performance analysis")
        print("   • These are ACTUAL predictions, not simulations")
        print("   • Green circles = Ground truth landmarks")
        print("   • Red X marks = Predicted landmarks")
        print("   • Yellow lines = Error vectors")
        
        plt.show()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Interactive Template Matching Viewer (Fixed)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of test samples to load (default: 10)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 TEMPLATE MATCHING INTERACTIVE VIEWER - FIXED VERSION")
    print("=" * 60)
    print("📊 Using REAL saved results with ~5.63 px error")
    print("=" * 60)
    
    viewer = TemplateLandmarkViewer()
    viewer.run_interactive_viewer(num_samples=args.samples)

if __name__ == "__main__":
    main()