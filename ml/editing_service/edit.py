#!/usr/bin/env python3
"""
Image editing script using the Qwen-Image-Edit model.

This script performs targeted image editing based on text prompts and mask regions
using the Qwen/Qwen-Image-Edit diffusion model.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from PIL import Image


def load_model(device: str = "cuda"):
    """
    Load the Qwen-Image-Edit model.
    
    Args:
        device: Device to run the model on ('cuda', 'cpu', 'mps')
        
    Returns:
        pipeline: The loaded QwenImageEditPipeline
    """
    try:
        from diffusers import QwenImageEditPipeline
        
        print("Loading Qwen-Image-Edit model...")
        print(f"Target device: {device}")
        
        # Load the pipeline
        pipeline = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32
        )
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            pipeline.to("cuda")
            print(f"Model loaded on CUDA (GPU: {torch.cuda.get_device_name(0)})")
        elif device == "mps" and torch.backends.mps.is_available():
            pipeline.to("mps")
            print("Model loaded on MPS (Apple Silicon)")
        else:
            if device != "cpu":
                print(f"Warning: {device} not available, falling back to CPU")
                print("Note: CPU inference will be significantly slower")
            pipeline.to("cpu")
            print("Model loaded on CPU")
        
        # Disable progress bar for cleaner output
        pipeline.set_progress_bar_config(disable=None)
        
        return pipeline
        
    except ImportError as e:
        print(f"Error: Missing required library. {e}")
        print("\nPlease install diffusers from git:")
        print("pip install git+https://github.com/huggingface/diffusers")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def load_and_prepare_images(
    original_path: str, 
    mask_path: str
) -> Tuple[Image.Image, Image.Image]:
    """
    Load and prepare the original image and mask.
    
    Args:
        original_path: Path to the original image
        mask_path: Path to the mask image
        
    Returns:
        Tuple of (original_image, mask_image)
    """
    try:
        # Load original image
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original image not found: {original_path}")
        
        original_image = Image.open(original_path).convert("RGB")
        print(f"Loaded original image: {original_path} (size: {original_image.size})")
        
        # Load mask image
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        
        mask_image = Image.open(mask_path).convert("L")  # Convert to grayscale
        print(f"Loaded mask image: {mask_path} (size: {mask_image.size})")
        
        # Resize mask to match original if needed
        if mask_image.size != original_image.size:
            print(f"Resizing mask from {mask_image.size} to {original_image.size}")
            mask_image = mask_image.resize(original_image.size, Image.LANCZOS)
        
        # Ensure mask is binary (black and white)
        mask_array = np.array(mask_image)
        mask_array = (mask_array > 127).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array, mode='L')
        
        return original_image, mask_image
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading images: {e}")
        sys.exit(1)


def perform_edit(
    pipeline,
    original_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    seed: int = 0,
    cfg_scale: float = 4.0,
    num_steps: int = 50
) -> Image.Image:
    """
    Perform the image edit using the model.
    
    Args:
        pipeline: The loaded QwenImageEditPipeline
        original_image: Original RGB image
        mask_image: Binary mask (white = edit region)
        prompt: Text description of desired edit
        seed: Random seed for reproducibility
        cfg_scale: Guidance scale (higher = stronger prompt adherence)
        num_steps: Number of inference steps
        
    Returns:
        Edited image
    """
    print(f"\nPerforming edit with prompt: '{prompt}'")
    print(f"Parameters: seed={seed}, cfg_scale={cfg_scale}, steps={num_steps}")
    
    # Prepare inputs
    inputs = {
        "image": original_image,
        "mask": mask_image,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": cfg_scale,
        "negative_prompt": " ",  # Minimal negative prompt as recommended
        "num_inference_steps": num_steps,
    }
    
    # Run inference
    try:
        with torch.inference_mode():
            print("Running inference...")
            output = pipeline(**inputs)
            edited_image = output.images[0]
            print("Edit completed successfully!")
            return edited_image
            
    except torch.cuda.OutOfMemoryError:
        print("Error: Out of GPU memory!")
        print("Try reducing image size or using CPU mode (--device cpu)")
        sys.exit(1)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


def main():
    """Main entry point for the image editing script."""
    parser = argparse.ArgumentParser(
        description='Edit images using Qwen-Image-Edit model with masks and prompts'
    )
    parser.add_argument(
        '--original_image',
        type=str,
        required=True,
        help='Path to the original image'
    )
    parser.add_argument(
        '--mask_image',
        type=str,
        required=True,
        help='Path to the mask image (white = edit region)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Text prompt describing the desired edit'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for edited image (default: <original_name>_edited.png)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on (default: cuda)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducibility (default: 0)'
    )
    parser.add_argument(
        '--cfg_scale',
        type=float,
        default=4.0,
        help='Guidance scale (default: 4.0)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Number of inference steps (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = 'cpu'
    
    # Memory check for CUDA
    if args.device == 'cuda':
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available VRAM: {vram_gb:.1f} GB")
        if vram_gb < 16:
            print("Warning: Less than 16GB VRAM detected. May encounter memory issues.")
            print("Consider using smaller images or CPU mode.")
    
    # Load model
    pipeline = load_model(args.device)
    
    # Load and prepare images
    original_image, mask_image = load_and_prepare_images(
        args.original_image, 
        args.mask_image
    )
    
    # Perform edit
    edited_image = perform_edit(
        pipeline,
        original_image,
        mask_image,
        args.prompt,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_steps=args.steps
    )
    
    # Save output
    if args.output is None:
        # Generate default output path
        original_path = Path(args.original_image)
        output_path = original_path.parent / f"{original_path.stem}_edited.png"
    else:
        output_path = Path(args.output)
    
    edited_image.save(output_path)
    print(f"\nEdited image saved to: {output_path.absolute()}")
    
    # Memory cleanup
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    
    print("Image editing completed successfully!")


if __name__ == "__main__":
    main()