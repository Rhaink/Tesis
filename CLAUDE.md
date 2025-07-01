# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical image analysis project for lung morphology analysis and COVID-19 detection from chest X-rays using Active Shape Models (ASM), Template Matching with PCA/Eigenpatches, and Convolutional Neural Networks.

## Key Architecture Components

### Core ASM Implementation
The Active Shape Model implementation is built around three main components:

1. **ShapeModel** (`pulmones/src/core/shape_model.py`): Implements Statistical Shape Models using Generalized Procrustes Analysis (GPA) and PCA to model shape variations
2. **AppearanceModel** (`pulmones/src/core/appearance_model.py`): Contains `AppearanceModel` for single-level and `MultiLevelAppearanceModel` for multi-resolution appearance modeling using image gradients
3. **ASMFitter** (`pulmones/src/core/asm_fitter.py`): Performs iterative shape fitting using combined shape and appearance models

### Template Matching with PCA/Eigenpatches Implementation
Alternative landmark detection method in `template_matching/` directory:

1. **EigenpatchesModel** (`template_matching/src/core/eigenpatches.py`): Implements PCA-based template matching using eigenpatches
2. **TemplateLandmarkPredictor** (`template_matching/src/core/landmark_predictor.py`): Multi-scale landmark prediction with geometric constraints
3. **Performance**: Achieves **5.63 ± 1.03 pixels** average error on test set (159 images)

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
source pulmones/.venv/bin/activate  # IMPORTANT: Always activate virtual environment
pip install -r requirements.txt
```

### Training ASM Models
```bash
cd /home/donrobot/Projects/Tesis
python pulmones/scripts/train_full_augmentation_asm.py
```

### Template Matching Commands
```bash
# Train template matching model
python template_matching/scripts/train_eigenpatches.py

# Process all test images (159 images)
python template_matching/scripts/process_all_images.py --dataset coordenadas_prueba_1.csv

# Generate visualizations for all test images
python template_matching/scripts/visualize_all_test_images.py

# Interactive viewer with real results (~5.63px error)
python template_matching/scripts/interactive_viewer.py --samples 20

# Quick process with menu
python template_matching/scripts/quick_process.py
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
- **CRITICAL**: When visualizing, scale landmarks to actual image size (typically 299×299)
- Scaling factor: `scale_x = image_width / 64.0`, `scale_y = image_height / 64.0`
- Profile sampling uses image-level coordinates after scaling

### Lung Landmark Connectivity (15 landmarks)
Anatomically correct connections for visualization:
- **Contour connections**: `[(0,12), (12,3), (3,5), (5,7), (7,14), (14,1), (1,13), (13,6), (6,4), (4,2), (2,11), (11,0)]`
- **Mediastinal connections**: `[(0,8), (8,9), (9,10), (10,1)]`
- **IMPORTANT**: Do NOT connect landmarks sequentially (0→1→2→3...), use anatomical order above

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
- **tqdm**: Progress bars for long operations
- **seaborn** (optional): Enhanced statistical visualizations

## Dataset Information

### Available Coordinate Files
Key datasets in `coordenadas/` directory:
- **coordenadas_entrenamiento_1.csv**: ~639 training images
- **coordenadas_prueba_1.csv**: ~159 test images (used for evaluations)
- **coordenadas_maestro_1.csv**: ~998 complete dataset
- **coordenadas_aligned_*.csv**: Pre-aligned versions using Procrustes

### CSV Format Issues
- **IMPORTANT**: Most coordinate CSV files have NO header row
- First line contains data, not column names
- When using `pd.read_csv()`, use `header=None` parameter
- Each row: 30 coordinate values (x1,y1,x2,y2,...,x15,y15) + image name

### Image Dataset Structure
```
COVID-19_Radiography_Dataset/
├── COVID/images/        # COVID-19 positive cases
├── Normal/images/       # Healthy control cases
└── Viral Pneumonia/images/  # Other viral pneumonia cases
```

## Template Matching Results

### Performance Metrics
- **Test set**: 159 images from `coordenadas_prueba_1.csv`
- **Average error**: 5.63 ± 1.03 pixels
- **Best result**: 3.41 pixels (Normal-1756)
- **Worst result**: 9.16 pixels (Viral Pneumonia-1092)
- **By pathology**:
  - Normal: 5.50 ± 1.06 px (72 images)
  - COVID-19: 5.76 ± 1.02 px (39 images)  
  - Viral Pneumonia: 5.71 ± 0.98 px (48 images)

### Key Files Generated
```
template_matching/
├── models/                    # Trained models
│   ├── landmark_predictor_*.pkl
│   └── ...
├── results/                   # Prediction results
│   └── results_coordenadas_prueba_1.pkl  # All 159 predictions
└── visualizations/           # Generated visualizations
    ├── all_test_images/      # Visualizations for all 159 images
    │   ├── landmark_predictions/   # Landmark overlays
    │   └── lung_contours/         # Anatomical contours
    └── ...
```

## Common Issues and Solutions

### Issue: Interactive viewer shows different error than reported
**Solution**: Ensure viewer loads saved results from `results_coordenadas_prueba_1.pkl` instead of using simulation

### Issue: Landmarks appear in wrong position  
**Solution**: Scale landmarks from 64×64 to actual image size using:
```python
scale_x = image_width / 64.0
scale_y = image_height / 64.0
scaled_landmarks[:, 0] *= scale_x
scaled_landmarks[:, 1] *= scale_y
```

### Issue: Contour lines crossing incorrectly
**Solution**: Use anatomical connectivity order, not sequential (see Lung Landmark Connectivity section)

### Issue: Missing first image when loading CSV
**Solution**: Use `pd.read_csv(file, header=None)` as files don't have headers