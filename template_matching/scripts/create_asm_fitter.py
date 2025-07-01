#!/usr/bin/env python3
"""
Create ASM Fitter from existing trained models.
"""

import sys
import os
import pickle
import logging

# Setup paths
PROJECT_ROOT_DIR = "/home/donrobot/Projects/Tesis"
SRC_DIR_PULMONES = os.path.join(PROJECT_ROOT_DIR, "pulmones", "src")
sys.path.insert(0, SRC_DIR_PULMONES)

from core.shape_model import ShapeModel
from core.appearance_model import MultiLevelAppearanceModel
from core.asm_fitter import ASMFitter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_asm_fitter():
    """Create ASMFitter from existing trained models."""
    models_dir = os.path.join(PROJECT_ROOT_DIR, 'pulmones', 'models')
    
    try:
        # Load shape model
        shape_model_path = os.path.join(models_dir, 'lung_shape_model_full_aug.pkl')
        logging.info(f"Loading shape model from: {shape_model_path}")
        
        # Load shape model using class method
        shape_model = ShapeModel.load(shape_model_path)
        
        logging.info(f"✓ Shape model loaded: {type(shape_model)}")
        logging.info(f"  Number of landmarks: {shape_model.num_landmarks}")
        logging.info(f"  Is trained: {shape_model._is_trained}")
        
        # Load appearance model (remove _meta.pkl suffix for the load method)
        app_model_base_path = os.path.join(models_dir, 'lung_appearance_model_full_aug')
        logging.info(f"Loading appearance model from: {app_model_base_path}")
        
        # Load appearance model using class method  
        appearance_model = MultiLevelAppearanceModel.load(app_model_base_path)
        
        logging.info(f"✓ Appearance model loaded: {type(appearance_model)}")
        logging.info(f"  Number of levels: {appearance_model.num_levels}")
        logging.info(f"  Is trained: {appearance_model._is_trained}")
        
        # Create ASM Fitter
        logging.info("Creating ASM Fitter...")
        asm_fitter = ASMFitter(shape_model, appearance_model)
        
        logging.info(f"✓ ASM Fitter created successfully")
        
        # Save the complete ASM fitter
        output_path = os.path.join(models_dir, 'full_augmentation_asm_fitter.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(asm_fitter, f)
        
        logging.info(f"✓ ASM Fitter saved to: {output_path}")
        
        return asm_fitter
        
    except Exception as e:
        logging.error(f"✗ Error creating ASM Fitter: {str(e)}")
        return None

def test_asm_fitter(asm_fitter):
    """Test the created ASM fitter."""
    try:
        logging.info("Testing ASM Fitter...")
        
        # Create a dummy image and initial shape for testing
        import numpy as np
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Get mean shape as initial shape  
        initial_shape = asm_fitter.shape_model.get_mean_shape_procrustes().copy()
        
        # Scale and center the shape for the image
        h, w = dummy_image.shape
        initial_shape[:, 0] = initial_shape[:, 0] * w / 64.0 + w / 2
        initial_shape[:, 1] = initial_shape[:, 1] * h / 64.0 + h / 2
        
        logging.info(f"  Initial shape: {initial_shape.shape}")
        logging.info(f"  Image shape: {dummy_image.shape}")
        
        # Try fitting (this might not converge well on random data, but should not crash)
        final_shape, converged = asm_fitter.fit_model_to_image(dummy_image, initial_shape)
        
        logging.info(f"✓ ASM fitting completed")
        logging.info(f"  Final shape: {final_shape.shape}")
        logging.info(f"  Converged: {converged}")
        
        return True
        
    except Exception as e:
        logging.error(f"✗ Error testing ASM Fitter: {str(e)}")
        return False

def main():
    """Main function."""
    logging.info("=== Creating ASM Fitter ===")
    
    # Create ASM fitter
    asm_fitter = create_asm_fitter()
    
    if asm_fitter is None:
        logging.error("Failed to create ASM Fitter")
        return
    
    # Test the fitter
    success = test_asm_fitter(asm_fitter)
    
    if success:
        logging.info("🎉 ASM Fitter created and tested successfully!")
        logging.info("The complete ASM model is now ready for comparison with template matching.")
    else:
        logging.error("❌ ASM Fitter creation failed during testing")

if __name__ == "__main__":
    main()