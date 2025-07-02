# Template Matching for Landmark Detection

This project implements an alternative approach to Active Shape Models (ASM) for lung landmark detection using eigenpatches-based template matching with PCA.

## Overview

The template matching approach uses Principal Component Analysis (PCA) to create compact representations of image patches around landmarks (eigenpatches). This method provides an alternative to ASM for landmark detection with the following advantages:

- **Direct template matching**: No iterative shape fitting required
- **Multi-scale search**: Coarse-to-fine detection using image pyramids  
- **Geometric constraints**: Shape statistics ensure anatomically plausible results
- **Ensemble prediction**: Multiple models can be combined for robustness

## Core Components

### 1. EigenpatchesModel (`src/core/eigenpatches.py`)
- **Single-scale eigenpatches**: PCA-based template matching for landmarks
- **Multi-scale eigenpatches**: Hierarchical search using image pyramids
- **Patch extraction**: Robust patch sampling with boundary handling
- **Template scoring**: Reconstruction error-based similarity scoring

### 2. TemplateLandmarkPredictor (`src/core/landmark_predictor.py`)
- **Complete predictor**: Combines eigenpatches with geometric constraints
- **Shape constraints**: Uses Procrustes analysis and PCA for shape modeling
- **Iterative refinement**: Alternates between template matching and shape fitting
- **Confidence scoring**: Provides reliability estimates for predictions

### 3. Evaluation Tools (`src/utils/evaluation.py`)
- **LandmarkEvaluator**: Comprehensive evaluation metrics and visualizations
- **MethodComparator**: Direct comparison between template matching and ASM
- **Statistical analysis**: Significance testing and error distribution analysis

## Key Features

### Template Matching Process
1. **Training Phase**:
   - Extract patches around ground truth landmarks
   - Normalize patches and compute PCA models per landmark
   - Learn shape statistics using Procrustes analysis

2. **Prediction Phase**:
   - Multi-scale sliding window search
   - Template matching using reconstruction error
   - Geometric constraint application
   - Iterative refinement

### Advantages over ASM
- **No initialization dependency**: Can work without good initial shape estimates
- **Robust to local minima**: Global search reduces fitting failures
- **Faster convergence**: Typically requires fewer iterations
- **Direct patch modeling**: No need for profile sampling along normals

## Usage

### Training Models

```bash
# Train eigenpatches model
cd /home/donrobot/Projects/Tesis
python template_matching/scripts/train_eigenpatches.py \
    --model_type predictor \
    --patch_size 21 \
    --n_components 20 \
    --pyramid_levels 3

# Train single-scale model only
python template_matching/scripts/train_eigenpatches.py \
    --model_type single \
    --patch_size 21 \
    --n_components 20

# Train multi-scale model only  
python template_matching/scripts/train_eigenpatches.py \
    --model_type multiscale \
    --patch_size 21 \
    --n_components 20 \
    --pyramid_levels 3
```

### Evaluation and Analysis

```bash
# Run comprehensive per-landmark evaluation
python template_matching/scripts/per_landmark_evaluation.py

# Display evaluation summary
python template_matching/scripts/show_evaluation_summary.py

# Compare template matching vs ASM
python template_matching/scripts/compare_methods.py \
    --template_model template_matching/models/landmark_predictor.pkl \
    --asm_model pulmones/models/full_augmentation_asm_fitter.pkl \
    --output_dir template_matching/evaluation
```

### Using the Models

```python
from template_matching.src.core.landmark_predictor import TemplateLandmarkPredictor
import cv2

# Load trained model
predictor = TemplateLandmarkPredictor()
predictor.load_model('template_matching/models/landmark_predictor.pkl')

# Predict landmarks
image = cv2.imread('path/to/image.png', cv2.IMREAD_GRAYSCALE)
result = predictor.predict_with_confidence(image)

landmarks = result['landmarks']
confidence = result['confidence_scores']
print(f"Mean confidence: {result['mean_confidence']:.3f}")
```

## Model Parameters

### Patch Size
- **Small patches (11x11)**: Fast but less distinctive
- **Medium patches (21x21)**: Good balance (recommended)
- **Large patches (31x31)**: More distinctive but slower

### PCA Components
- **Few components (5-10)**: Fast but may lose important variations
- **Medium components (15-25)**: Good balance (recommended)  
- **Many components (30+)**: More detailed but prone to overfitting

### Pyramid Levels
- **Single scale**: Fastest but may miss targets
- **2-3 levels**: Good balance (recommended)
- **4+ levels**: More thorough but slower

## File Structure

```
template_matching/
├── src/
│   ├── core/
│   │   ├── eigenpatches.py          # PCA-based patch models
│   │   └── landmark_predictor.py    # Complete prediction system
│   └── utils/
│       └── evaluation.py            # Evaluation and comparison tools
├── scripts/
│   ├── train_eigenpatches.py        # Training script
│   └── compare_methods.py           # Method comparison script
├── models/                          # Trained models directory
├── notebooks/                       # Jupyter notebooks for analysis
└── README.md                        # This file
```

## Performance Results

### Overall Performance
- **Test Dataset**: 159 images (72 Normal, 48 Viral Pneumonia, 39 COVID)
- **Average Error**: **5.628 ± 0.190 pixels** across all 15 landmarks
- **Validation**: Matches documented performance (5.63 px) with 99.97% accuracy

### Per-Landmark Performance Rankings

**🏆 Best Performing Landmarks:**
1. **L11 (Right_Lower_Edge)**: 5.297 ± 3.016 px
2. **L14 (Left_Upper)**: 5.408 ± 2.792 px (most consistent)
3. **L5 (Right_Mid)**: 5.449 ± 2.805 px

**⚠️ Challenging Landmarks:**
- **L9 (Center_Medial)**: 5.995 ± 3.158 px (highest error)
- **L12 (Right_Upper_Mid)**: 5.950 ± 3.434 px (most variable)
- **L8 (Right_Medial_Top)**: 5.757 ± 3.241 px

### Pathology Performance
- **Normal**: 5.501 ± 0.262 px (72 samples)
- **Viral Pneumonia**: 5.708 ± 0.389 px (48 samples)
- **COVID**: 5.764 ± 0.388 px (39 samples)

*COVID cases show 4.8% higher error than Normal cases*

### Clinical Insights
- **Edge landmarks** outperform mediastinal landmarks (clearer boundaries)
- **Sub-6 pixel accuracy** achieved for all landmarks
- **Consistent performance** across different pathologies
- **Clinically acceptable** precision for lung morphology analysis

## Evaluation Metrics

The evaluation system provides comprehensive analysis:

- **Point-to-point error**: Mean Euclidean distance between predicted and ground truth landmarks
- **Cumulative Error Distribution (CED)**: Proportion of samples below error thresholds
- **Per-landmark analysis**: Individual landmark performance statistics with pathology breakdown
- **Statistical significance**: Wilcoxon signed-rank test for method comparison
- **Anatomical region analysis**: Performance by lung region
- **Consistency metrics**: Standard deviation and error range analysis

## Dependencies

The template matching system requires:
- numpy
- opencv-python  
- scikit-learn
- scipy
- matplotlib
- seaborn
- pandas

## Integration with Main Project

This template matching system is designed to integrate with the main lung morphology analysis project:

- Uses the same coordinate system and data format as ASM
- Compatible with existing evaluation pipelines
- Can be used as a drop-in replacement for ASM initialization
- Supports the same image preprocessing and scaling

## Future Enhancements

Potential improvements include:
- **Deep learning patches**: Replace PCA with learned features
- **Adaptive patch sizes**: Variable patch sizes per landmark
- **Robust optimization**: Better handling of outliers and occlusions
- **Real-time inference**: Optimized implementation for clinical use