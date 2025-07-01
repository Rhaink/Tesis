# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical image analysis project for lung morphology analysis and COVID-19 detection from chest X-rays using Active Shape Models (ASM) and Convolutional Neural Networks.

## Key Architecture Components

### Core ASM Implementation
The Active Shape Model implementation is built around three main components:

1. **ShapeModel** (`pulmones/src/core/shape_model.py`): Implements Statistical Shape Models using Generalized Procrustes Analysis (GPA) and PCA to model shape variations
2. **AppearanceModel** (`pulmones/src/core/appearance_model.py`): Contains `AppearanceModel` for single-level and `MultiLevelAppearanceModel` for multi-resolution appearance modeling using image gradients
3. **ASMFitter** (`pulmones/src/core/asm_fitter.py`): Performs iterative shape fitting using combined shape and appearance models

### Data Flow Architecture
- **Landmarks**: Stored as CSV files in `coordenadas/` with x,y coordinates for lung contours
- **Images**: X-ray images organized by condition in `COVID-19_Radiography_Dataset/`
- **Indices**: Metadata CSV files in `indices/` linking images to coordinates and categories
- **Models**: Trained models saved as pickle files in `pulmones/models/`

### Key Algorithms
- **Procrustes Analysis**: Aligns shapes to remove translation, rotation, and scale variations
- **PCA Shape Space**: Projects aligned shapes into principal component space for compact representation
- **Multi-Level Search**: Uses image pyramids for coarse-to-fine shape fitting
- **Profile Matching**: Samples gradients along normals to landmark points for appearance matching

## Common Development Commands

### Environment Setup
```bash
cd /home/donrobot/Projects/Tesis
pip install -r requirements.txt
```

### Training ASM Models
```bash
cd /home/donrobot/Projects/Tesis
python pulmones/scripts/train_full_augmentation_asm.py
```

### Analyzing Dataset Morphology
```bash
cd /home/donrobot/Projects/Tesis
python pulmones/scripts/analyze_dataset_morphology.py
```

### Processing and Warping Images
```bash
cd /home/donrobot/Projects/Tesis
python pulmones/scripts/process_and_warp_all.py
```

### Running Visual Demo
```bash
cd /home/donrobot/Projects/Tesis
python pulmones/scripts/visual_demo.py
```

### Dataset Curation by Morphology
```bash
cd /home/donrobot/Projects/Tesis
python pulmones/scripts/curated_dataset_by_morphology.py
```

## Important Implementation Details

### Coordinate Systems
- Landmarks use a 64x64 reference coordinate system
- Scaling is applied when working with different image resolutions
- Profile sampling uses image-level coordinates after scaling

### File Paths
All scripts use absolute paths rooted at `/home/donrobot/Projects/Tesis`. When modifying paths:
- Update `PROJECT_ROOT_DIR` in script headers
- Ensure `SRC_DIR_PULMONES` is added to `sys.path` for imports

### Model Persistence
- Shape models save: mean shape, PCA model, number of landmarks
- Appearance models save: per-level, per-landmark statistics
- Multi-level models use a meta file plus separate files per pyramid level

### Data Augmentation
The training pipeline supports:
- Shape augmentation via PCA-based synthetic shape generation
- Intensity augmentation (contrast, brightness, noise, blur)
- Controlled variation within ±2 standard deviations

## Key Dependencies
- **numpy**: Core numerical operations
- **opencv-python**: Image processing and pyramids
- **scipy**: Statistical functions and optimizations
- **scikit-learn**: PCA implementation
- **matplotlib**: Visualization
- **numba**: JIT compilation for performance-critical code