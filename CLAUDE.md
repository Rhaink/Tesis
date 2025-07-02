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

### Delaunay Morphing Implementation (NEW)
Novel lung morphology approach combining Template Matching precision with ASM-style image warping:

1. **DelaunayLungMorpher** (`delaunay_morphing/src/core/delaunay_lung_morpher.py`): Implements Delaunay triangulation-based image morphing with anatomically correct landmark connectivity
2. **Key Features**: 
   - Uses exact Template Matching landmarks (5.63±1.03px error)
   - Performs piecewise affine transformations via Delaunay triangulation
   - Warps images to canonical/mean shape for morphological analysis
   - Maintains anatomical lung contour and mediastinal connections
3. **Results**: 159 complete visualizations showing TM landmarks, canonical morphing, and triangulation quality

### Matching Geometric Implementation (NEW)
Hybrid approach combining Template Matching precision with manual labeling geometric strategy:

1. **GeometricLandmarkPredictor** (`matching_geometric/src/core/geometric_predictor.py`): Implements hybrid TM + geometric construction method
2. **Key Features**:
   - Uses exact Template Matching for key points 0 and 1 (5.63px error model)
   - Applies quartile strategy from manual labeling methodology
   - Constructs geometric landmarks using perpendicular lines
   - Improves quartile point precision by 16.3% over pure Template Matching
3. **Performance**: 
   - **Quartile landmarks**: 4.868 ± 2.544 px (vs TM: 5.813 ± 1.954 px)
   - **Success rate**: 56.8% cases better than Template Matching
   - **Best improvement**: 21.7% better for middle quartile point
4. **Results**: 159 processed images with comparative analysis

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

# Run comprehensive per-landmark evaluation (NEW)
python template_matching/scripts/per_landmark_evaluation.py

# Display evaluation summary (NEW)
python template_matching/scripts/show_evaluation_summary.py
```

### Delaunay Morphing Commands (NEW)
```bash
# Process all 159 images with Delaunay morphing using Template Matching landmarks
python delaunay_morphing/process_all_159_images.py

# Generate CORRECT visualizations with exact TM landmarks (5.63±1.03px error)
python delaunay_morphing/fix_correct_tm_order.py

# Complete any remaining visualizations 
python delaunay_morphing/complete_remaining_correct.py

# View completed results
ls delaunay_morphing/correct_tm_visualizations/*.png | wc -l  # Should show 159
```

### Matching Geometric Commands (NEW)
```bash
# Process all 159 test images with geometric method
python matching_geometric/scripts/process_all_159_images.py

# Compare geometric vs Template Matching precision
python matching_geometric/scripts/compare_geometric_vs_tm.py

# Test geometric predictor on single image
python matching_geometric/scripts/test_geometric_predictor.py

# Verify coordinate accuracy with saved TM results
python matching_geometric/scripts/verify_coordinates.py

# Validate Template Matching error calculations
python matching_geometric/scripts/verify_tm_error.py

# Generate visual comparisons (3 methods side by side)
python matching_geometric/scripts/generate_all_comparisons.py

# Generate comparisons with anatomical contours (RECOMMENDED)
python matching_geometric/scripts/quick_contours_159.py

# View results
ls matching_geometric/visualizations/all_159_images/*.png | wc -l         # 159 basic
ls matching_geometric/visualizations/comparaciones_159/*.png | wc -l      # 159 comparisons  
ls matching_geometric/visualizations/contornos_rapidos_159/*.png | wc -l  # 159 with contours
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
- **Average error**: **5.628 ± 0.190 pixels** across all 15 landmarks
- **Validation**: Matches documented performance (5.63 px) with 99.97% accuracy
- **By pathology**:
  - Normal: 5.501 ± 0.262 px (72 images)
  - Viral Pneumonia: 5.708 ± 0.389 px (48 images)
  - COVID-19: 5.764 ± 0.388 px (39 images)

### Per-Landmark Performance
- **Best performing landmark**: L11 (Right_Lower_Edge) - 5.297 ± 3.016 px
- **Most consistent landmark**: L14 (Left_Upper) - 5.408 ± 2.792 px
- **Most challenging landmark**: L9 (Center_Medial) - 5.995 ± 3.158 px
- **Clinical insight**: Edge landmarks outperform mediastinal landmarks
- **All landmarks achieve sub-6 pixel accuracy**

### Key Files Generated
```
template_matching/
├── models/                    # Trained models
│   ├── landmark_predictor_*.pkl
│   └── ...
├── results/                   # Prediction results
│   └── results_coordenadas_prueba_1.pkl  # All 159 predictions
├── evaluation/               # *** NEW: Per-landmark evaluation results ***
│   ├── per_landmark_statistics.csv        # Detailed landmark statistics
│   ├── per_landmark_by_pathology.csv      # Pathology breakdown
│   ├── per_landmark_evaluation_report.txt # Comprehensive text report
│   ├── per_landmark_overall_analysis.png  # Overall analysis plots
│   ├── per_landmark_by_pathology.png      # Pathology comparison
│   ├── error_heatmaps.png                 # Error heatmap visualizations
│   ├── statistical_summary.png            # Statistical analysis
│   └── best_worst_landmarks.png           # Performance ranking plots
├── scripts/                  # *** UPDATED: New evaluation scripts ***
│   ├── per_landmark_evaluation.py         # Comprehensive evaluation
│   ├── show_evaluation_summary.py         # Quick summary display
│   └── ...
└── visualizations/           # Generated visualizations
    ├── all_test_images/      # Visualizations for all 159 images
    │   ├── landmark_predictions/   # Landmark overlays
    │   └── lung_contours/         # Anatomical contours
    └── ...
```

## Delaunay Morphing Results (NEW)

### Performance and Features
- **Input**: Exact Template Matching landmarks (5.63±1.03px error maintained)
- **Method**: Delaunay triangulation-based piecewise affine transformation
- **Output**: Images warped to canonical/mean shape for morphological analysis
- **Anatomical Correctness**: Maintains proper lung contour and mediastinal connections

### Key Files Generated
```
delaunay_morphing/
├── src/core/
│   └── delaunay_lung_morpher.py       # Core morphing engine
├── processed_159/
│   ├── canonical_shape.npy            # Mean shape from 159 TM predictions
│   └── tm_predictions_*.npy           # Processed landmark data
├── correct_tm_visualizations/         # *** 159 COMPLETE VISUALIZATIONS ***
│   ├── Normal_001_Normal-3173.png     # Shows: TM landmarks + Warped + Triangulation
│   ├── COVID_002_COVID-1652.png       # 3-panel visualization per image
│   └── ... (159 total files)
└── Scripts:
    ├── process_all_159_images.py      # Main processing pipeline
    ├── fix_correct_tm_order.py        # Generate correct visualizations
    └── complete_remaining_correct.py  # Complete any missing files
```

## Matching Geometric Results (NEW)

### Performance and Features
- **Input**: Template Matching points 0,1 (5.63px error) + geometric quartile construction
- **Method**: Hybrid TM precision + manual labeling geometric strategy
- **Output**: Improved quartile landmarks with 16.3% better precision
- **Validation**: Direct comparison with ground truth on 159 test images

### Key Files Generated
```
matching_geometric/
├── src/core/
│   └── geometric_predictor.py         # GeometricLandmarkPredictor class
├── scripts/
│   ├── process_all_159_images.py      # Process all test images
│   ├── compare_geometric_vs_tm.py     # Precision comparison analysis
│   ├── verify_coordinates.py          # Coordinate system validation
│   └── verify_tm_error.py            # TM error verification
├── visualizations/
│   ├── all_159_images/                # *** 159 BASIC VISUALIZATIONS ***
│   │   ├── Normal-3173.png           # Shows: Main line + quartile points  
│   │   └── ... (159 total files)
│   ├── comparaciones_159/             # *** 159 COMPARISON VISUALIZATIONS ***
│   │   ├── comparacion_001_Normal_3173.png  # 3 methods side by side
│   │   └── ... (159 total files)
│   ├── contornos_rapidos_159/        # *** 159 CONTOUR VISUALIZATIONS ⭐ ***
│   │   ├── contorno_001_Normal_3173.png     # With anatomical contours
│   │   └── ... (159 total files)
│   ├── geometric_vs_tm_comparison.png # Performance comparison charts
│   ├── method_comparison_grid.png     # Grid comparison visualization
│   ├── detailed_method_comparison.png # Detailed single image analysis
│   └── geometric_vs_tm_stats.csv      # Detailed statistical analysis
└── README.md                          # Complete method documentation
```

### Precision Improvement Results
- **Overall quartile improvement**: 16.3% better than Template Matching
- **Middle quartile**: 21.7% improvement (best performing point)
- **Success rate**: 56.8% of cases show geometric method superiority
- **Consistency**: Lower standard deviation in geometric predictions

### Visual Outputs Generated
- **Basic visualizations**: 159 images showing geometric method only
- **Comparison visualizations**: 159 images with 3 methods side by side
- **Contour visualizations**: 159 images with anatomical contours ⭐ **RECOMMENDED**
  - Ground Truth: Green contours with all anatomical connections
  - Template Matching: Red contours with 15 landmarks
  - Matching Geometric: Green main line + red key points + yellow quartiles
  - Processing speed: ~106 images/second

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