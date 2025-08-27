#!/usr/bin/env python3
"""
Clothing segmentation script using the SegFormer model.

This script segments clothing and body parts from images using the
sayeed99/segformer_b3_clothes model from Hugging Face.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from PIL import Image
import torch


def load_model():
    """
    Load the SegFormer model for clothes segmentation.
    
    Returns:
        pipeline: The loaded segmentation pipeline
    """
    try:
        from transformers import pipeline
        print("Loading segformer_b3_clothes model...")
        pipe = pipeline("image-segmentation", model="sayeed99/segformer_b3_clothes")
        print("Model loaded successfully!")
        return pipe
    except ImportError as e:
        print(f"Error: Missing required library. Please install transformers: {e}")
        print("You may need to install the dev version:")
        print("pip install git+https://github.com/huggingface/transformers.git")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def process_image(image_path: str, pipeline) -> Dict[str, Any]:
    """
    Process an image through the segmentation pipeline.
    
    Args:
        image_path: Path to the input image
        pipeline: The segmentation pipeline
        
    Returns:
        Dictionary containing segmentation results
    """
    try:
        # Load the image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run segmentation
        print(f"Processing image: {image_path}")
        results = pipeline(image)
        
        return {
            "image_size": image.size,
            "segments": results
        }
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


def create_visual_mask(segments: List[Dict], image_size: tuple) -> Image.Image:
    """
    Create a color-coded visual mask from segmentation results.
    
    Args:
        segments: List of segmentation results
        image_size: Size of the original image (width, height)
        
    Returns:
        PIL Image with color-coded segmentation mask
    """
    # Define colors for each clothing category
    color_map = {
        'Hat': (255, 0, 0),           # Red
        'Hair': (139, 69, 19),         # Brown
        'Sunglasses': (255, 255, 0),   # Yellow
        'Upper-clothes': (0, 128, 0),   # Green
        'Skirt': (255, 192, 203),      # Pink
        'Pants': (0, 0, 255),          # Blue
        'Dress': (128, 0, 128),        # Purple
        'Belt': (255, 165, 0),         # Orange
        'Left-shoe': (64, 64, 64),     # Dark Gray
        'Right-shoe': (128, 128, 128), # Gray
        'Face': (255, 218, 185),       # Peach
        'Left-leg': (255, 228, 196),   # Bisque
        'Right-leg': (250, 235, 215),  # Antique White
        'Left-arm': (245, 222, 179),   # Wheat
        'Right-arm': (238, 203, 173),  # Peach Puff
        'Bag': (160, 82, 45),          # Sienna
        'Scarf': (255, 0, 255),        # Magenta
        'Background': (192, 192, 192), # Light Gray
    }
    
    # Create base image
    mask_image = Image.new('RGB', image_size, (0, 0, 0))
    
    # Apply each segment with its corresponding color
    for segment in segments:
        label = segment.get('label', 'Unknown')
        mask = segment.get('mask')
        
        if mask is not None:
            # Convert mask to PIL Image if it's not already
            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(np.array(mask) * 255).convert('L')
            
            # Resize mask to match image size if needed
            if mask.size != image_size:
                mask = mask.resize(image_size, Image.NEAREST)
            
            # Get color for this label
            color = color_map.get(label, (128, 128, 128))
            
            # Create colored layer
            colored_layer = Image.new('RGB', image_size, color)
            
            # Apply mask to colored layer and composite
            mask_image.paste(colored_layer, (0, 0), mask)
    
    return mask_image


def save_outputs(segments: List[Dict], image_size: tuple, base_path: str) -> None:
    """
    Save segmentation outputs as visual mask and JSON data.
    
    Args:
        segments: Segmentation results
        image_size: Size of the original image
        base_path: Base path for output files (without extension)
    """
    # Create visual mask
    visual_mask = create_visual_mask(segments, image_size)
    mask_path = f"{base_path}_mask.png"
    visual_mask.save(mask_path)
    print(f"Visual mask saved to: {mask_path}")
    
    # Prepare JSON data
    json_data = {
        "image_size": {
            "width": image_size[0],
            "height": image_size[1]
        },
        "segments": []
    }
    
    for segment in segments:
        score = segment.get('score')
        segment_data = {
            "label": segment.get('label'),
            "score": float(score) if score is not None else 0.0
        }
        
        # Convert mask to list format for JSON storage
        mask = segment.get('mask')
        if mask is not None:
            if isinstance(mask, Image.Image):
                mask_array = np.array(mask)
            else:
                mask_array = np.array(mask)
            
            # Store mask as binary (0 or 1) values
            segment_data['mask'] = (mask_array > 0).astype(int).tolist()
        
        json_data['segments'].append(segment_data)
    
    # Save JSON data
    json_path = f"{base_path}_mask_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Segmentation data saved to: {json_path}")


def main():
    """Main entry point for the segmentation script."""
    parser = argparse.ArgumentParser(
        description='Segment clothing and body parts from images'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to the input image'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load model
    pipeline = load_model()
    
    # Process image
    results = process_image(args.image_path, pipeline)
    
    # Generate output base path
    image_path = Path(args.image_path)
    base_path = str(image_path.parent / image_path.stem)
    
    # Save outputs
    save_outputs(results['segments'], results['image_size'], base_path)
    
    print("Segmentation completed successfully!")


if __name__ == "__main__":
    main()