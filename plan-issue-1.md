# Implementation Plan - Issue #1: Clothes Segmentation Model

## Issue Link
https://github.com/navv42/style-up/issues/1

## Objective
Create a standalone Python script that uses the `sayeed99/segformer_b3_clothes` model to identify and segment clothing and body parts from images.

## Input/Output Specification

### Input
- Command-line argument: `--image_path` pointing to an image file
- Supported formats: JPEG, PNG, WEBP

### Output
1. **Visual segmentation mask** (`<image_name>_mask.png`): Color-coded visualization of segmented regions
2. **JSON data file** (`<image_name>_mask_data.json`): Raw segmentation data for downstream tasks

## Implementation Steps

### 1. Setup Project Structure
- Create `ml/segmentation_service/` directory
- Create `segment.py` script
- Create `requirements.txt` file

### 2. Core Implementation Components

#### Model Loading
- Use the pipeline approach for simplicity:
  ```python
  from transformers import pipeline
  pipe = pipeline("image-segmentation", model="sayeed99/segformer_b3_clothes")
  ```

#### Image Processing
1. Load input image using PIL
2. Run inference through the pipeline
3. Extract segmentation masks for each detected category

#### Output Generation
1. **Visual Mask**: Create a color-coded image where each segment has a distinct color
2. **JSON Data**: Store segmentation information including:
   - Detected categories
   - Mask arrays (as lists)
   - Confidence scores if available

### 3. Error Handling
- Invalid file path: Clear error message and exit gracefully
- Unsupported image format: Inform user of supported formats
- Model loading failures: Suggest installing dev version of transformers if needed
- GPU/CUDA errors: Fallback to CPU processing

### 4. Resource Management
- Model will be downloaded on first run (~250MB)
- Use CPU by default for compatibility
- Add optional `--device` flag for GPU usage
- Memory usage: ~2-4GB RAM expected

### 5. Testing Strategy
- Test with both provided images: `girl.jpg.webp` and `girl_belt.webp`
- Verify visual masks show distinct colors for different clothing segments
- Validate JSON structure is consistent and parseable
- Test edge cases: missing file, corrupted image

## Dependencies
- transformers (latest or dev version)
- torch
- torchvision
- Pillow
- numpy
- argparse

## Success Criteria
✅ Script runs from command line with image path argument
✅ Loads segformer_b3_clothes model successfully
✅ Generates visual segmentation mask
✅ Outputs JSON data file
✅ Works with all test images
✅ Includes proper error handling
✅ Has requirements.txt file