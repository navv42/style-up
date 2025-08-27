# Implementation Plan - Issue #2: Generative Image Editing Model

## Issue Link
https://github.com/navv42/style-up/issues/2

## Objective
Create a standalone Python script that uses the `Qwen/Qwen-Image-Edit` model to perform targeted image editing based on text prompts and mask regions.

## Input/Output Specification

### Input
- `--original_image`: Path to the source image
- `--mask_image`: Path to a black-and-white mask (white = edit region)
- `--prompt`: Text description of desired change

### Output
- Edited image with changes applied only to masked region
- Filename: `<original_name>_edited.png`

## Implementation Steps

### 1. Setup Project Structure
- Create `ml/editing_service/` directory
- Create `edit.py` script
- Create `requirements.txt` file
- Create `README.md` with hardware requirements

### 2. Core Implementation Components

#### Model Loading
- Use QwenImageEditPipeline from diffusers
- Configure for optimal performance (bfloat16, appropriate device)
- Handle model download on first run

#### Image Processing
1. Load original image and convert to RGB
2. Load mask image and ensure it's black/white
3. Apply mask to constrain editing region
4. Run inference with prompt
5. Save edited result

#### Key Parameters
- `true_cfg_scale`: 4.0 (guidance strength)
- `num_inference_steps`: 50 (quality/speed tradeoff)
- `negative_prompt`: " " (minimal negative guidance)

### 3. Error Handling
- Invalid file paths
- Unsupported image formats
- GPU memory errors (fallback to CPU if possible)
- Model loading failures

### 4. Resource Management
- **GPU Requirements**: >20GB VRAM recommended
- Model size: ~10-15GB
- Fallback to CPU mode with warning
- Memory optimization with torch.inference_mode()

### 5. Testing Strategy
1. Create manual test mask for clothing region
2. Test with different prompts:
   - "a red leather jacket"
   - "a blue denim shirt"
   - "a floral dress"
3. Verify edits stay within masked region
4. Test edge cases

## Dependencies
- diffusers (latest from git)
- torch
- torchvision
- Pillow
- numpy
- accelerate (for GPU optimization)

## Hardware Requirements
- **Recommended**: GPU with 20+ GB VRAM (e.g., RTX 3090, A100)
- **Minimum**: 16GB system RAM for CPU mode (very slow)
- **Storage**: ~15GB for model weights

## Success Criteria
✅ Script accepts three command-line arguments
✅ Loads Qwen-Image-Edit model successfully
✅ Generates edited image with changes only in masked region
✅ Includes requirements.txt
✅ Includes README.md with hardware requirements
✅ Successfully tests with manually created mask