#!/usr/bin/env python3
"""
Image editing script using the Qwen-Image-Edit model via Hugging Face API.

This script performs targeted image editing using the HF InferenceClient,
which is much lighter than downloading the full model locally.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_client():
    """
    Initialize the Hugging Face InferenceClient.
    
    Returns:
        InferenceClient configured for image editing
    """
    try:
        from huggingface_hub import InferenceClient
        
        # Get API key from environment
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            print("Error: HF_TOKEN not found in environment variables")
            print("Please set HF_TOKEN in your .env file or environment")
            sys.exit(1)
        
        print("Initializing Hugging Face InferenceClient...")
        client = InferenceClient(
            provider="fal-ai",
            api_key=api_key,
        )
        print("Client initialized successfully!")
        return client
        
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Please install it with: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing client: {e}")
        sys.exit(1)


def apply_mask_to_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Apply mask to image to create a composite for the API.
    
    The API expects a single image input, so we need to somehow indicate
    the region to edit. One approach is to blend or mark the region.
    
    Args:
        image: Original image
        mask: Binary mask (white = edit region)
        
    Returns:
        Composite image for API input
    """
    # Convert images to arrays
    img_array = np.array(image)
    mask_array = np.array(mask.convert('L'))
    
    # Normalize mask to 0-1 range
    mask_norm = mask_array / 255.0
    
    # Create a subtle overlay to indicate edit region
    # We'll slightly darken the non-edit regions
    composite = img_array.copy()
    for c in range(3):
        composite[:, :, c] = (
            img_array[:, :, c] * (0.7 + 0.3 * mask_norm)
        ).astype(np.uint8)
    
    return Image.fromarray(composite)


def perform_edit_api(
    client,
    original_image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
    output_path: str
) -> Image.Image:
    """
    Perform image editing using the HF InferenceClient API.
    
    Args:
        client: Initialized InferenceClient
        original_image: Original RGB image
        mask_image: Binary mask (white = edit region)
        prompt: Text description of desired edit
        output_path: Path to save the edited image
        
    Returns:
        Edited image
    """
    print(f"\nPerforming edit with prompt: '{prompt}'")
    print("Sending request to Hugging Face API...")
    
    try:
        # Apply mask to create input image
        # Note: The API might not directly support masks, 
        # so we may need to be creative with the prompt
        input_image = apply_mask_to_image(original_image, mask_image)
        
        # Save input image to bytes
        from io import BytesIO
        img_bytes = BytesIO()
        input_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Make API call
        edited_image = client.image_to_image(
            img_bytes,
            prompt=prompt,
            model="Qwen/Qwen-Image-Edit",
        )
        
        print("Edit completed successfully!")
        
        # Save the result
        edited_image.save(output_path)
        print(f"Edited image saved to: {output_path}")
        
        return edited_image
        
    except Exception as e:
        print(f"Error during API call: {e}")
        print("\nPossible issues:")
        print("- Check your HF_TOKEN is valid")
        print("- Ensure you have API access to the model")
        print("- Check your internet connection")
        print("- The model might be loading (try again in a moment)")
        sys.exit(1)


def load_and_prepare_images(
    original_path: str, 
    mask_path: str
) -> tuple[Image.Image, Image.Image]:
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


def main():
    """Main entry point for the API-based image editing script."""
    parser = argparse.ArgumentParser(
        description='Edit images using Qwen-Image-Edit model via HF API'
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
        help='Output path for edited image (default: <original_name>_edited_api.png)'
    )
    
    args = parser.parse_args()
    
    # Check for API token
    if not os.environ.get("HF_TOKEN"):
        print("Error: HF_TOKEN not found!")
        print("\nPlease either:")
        print("1. Create a .env file with: HF_TOKEN=your_token_here")
        print("2. Or set the environment variable: export HF_TOKEN=your_token_here")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Initialize client
    client = initialize_client()
    
    # Load images
    original_image, mask_image = load_and_prepare_images(
        args.original_image, 
        args.mask_image
    )
    
    # Generate output path
    if args.output is None:
        original_path = Path(args.original_image)
        output_path = str(original_path.parent / f"{original_path.stem}_edited_api.png")
    else:
        output_path = args.output
    
    # Perform edit
    perform_edit_api(
        client,
        original_image,
        mask_image,
        args.prompt,
        output_path
    )
    
    print("\nImage editing completed successfully!")
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()