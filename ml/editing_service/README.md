# Image Editing Service

This service uses the Qwen-Image-Edit model to perform targeted image editing based on text prompts and mask regions.

## Hardware Requirements

### Recommended
- **GPU**: NVIDIA GPU with 20+ GB VRAM (e.g., RTX 3090, RTX 4090, A100)
- **RAM**: 32GB system memory
- **Storage**: ~15GB for model weights

### Minimum
- **GPU**: NVIDIA GPU with 16GB VRAM (may require smaller images)
- **RAM**: 16GB system memory
- **Storage**: ~15GB for model weights

### CPU Mode (Not Recommended)
- **RAM**: 32GB+ system memory
- **Note**: CPU inference is extremely slow (10-20x slower than GPU)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The requirements include installing diffusers from git to ensure latest version compatibility.

## Usage

Basic usage:
```bash
python edit.py \
    --original_image ../test_images/person1.jpg \
    --mask_image ../test_images/person1_shirt_mask.png \
    --prompt "a red leather jacket"
```

Advanced options:
```bash
python edit.py \
    --original_image input.jpg \
    --mask_image mask.png \
    --prompt "a blue denim shirt" \
    --output custom_output.png \
    --device cuda \
    --seed 42 \
    --cfg_scale 4.0 \
    --steps 50
```

### Arguments

- `--original_image`: Path to the source image
- `--mask_image`: Path to black-and-white mask (white areas will be edited)
- `--prompt`: Text description of the desired change
- `--output`: (Optional) Output path for edited image
- `--device`: (Optional) Device to run on: cuda, mps, or cpu (default: cuda)
- `--seed`: (Optional) Random seed for reproducibility (default: 0)
- `--cfg_scale`: (Optional) Guidance scale, higher = stronger prompt adherence (default: 4.0)
- `--steps`: (Optional) Number of inference steps, more = better quality but slower (default: 50)

## Creating Masks

Masks should be black and white images where:
- **White pixels (255)**: Areas to be edited
- **Black pixels (0)**: Areas to preserve

You can create masks using:
- Image editing software (GIMP, Photoshop)
- Python with PIL/OpenCV
- Output from the segmentation service

## Performance Notes

### Inference Time (per image)
- **RTX 3090/4090**: ~10-20 seconds
- **RTX 3060 (12GB)**: ~30-40 seconds
- **Apple M1/M2 (MPS)**: ~40-60 seconds
- **CPU**: 5-10+ minutes

### Memory Usage
- Model loading: ~10-15GB VRAM
- Inference: Additional 5-10GB VRAM depending on image size
- Larger images require more memory

### Tips for Memory Optimization
1. Use smaller images (512x512 or 768x768)
2. Reduce batch size to 1 (default)
3. Use lower precision (bfloat16, enabled by default on GPU)
4. Clear cache between runs if processing multiple images

## Troubleshooting

### Out of Memory Error
- Reduce image size
- Use CPU mode (very slow): `--device cpu`
- Close other GPU applications

### Model Download Issues
- Ensure stable internet connection
- Model will be cached after first download (~15GB)
- Cache location: `~/.cache/huggingface/hub/`

### Poor Edit Quality
- Ensure mask accurately covers desired region
- Try different prompts (be specific)
- Adjust cfg_scale (3.0-7.0 range)
- Increase steps for better quality (75-100)

## Examples

### Change clothing color
```bash
python edit.py \
    --original_image person.jpg \
    --mask_image shirt_mask.png \
    --prompt "bright yellow shirt"
```

### Add patterns or textures
```bash
python edit.py \
    --original_image person.jpg \
    --mask_image dress_mask.png \
    --prompt "floral pattern dress with roses"
```

### Change material
```bash
python edit.py \
    --original_image person.jpg \
    --mask_image jacket_mask.png \
    --prompt "shiny metallic silver jacket"
```