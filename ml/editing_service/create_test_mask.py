#!/usr/bin/env python3
"""
Helper script to create a test mask from segmentation output.
"""

import json
import numpy as np
from PIL import Image
import sys

def create_mask_from_segmentation(json_path: str, output_path: str, target_label: str = "Upper-clothes"):
    """Create a binary mask from segmentation JSON data."""
    
    # Load segmentation data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    width = data['image_size']['width']
    height = data['image_size']['height']
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Find target segment and apply mask
    for segment in data['segments']:
        if segment['label'] == target_label:
            segment_mask = np.array(segment['mask'])
            mask[segment_mask > 0] = 255
            print(f"Found {target_label} segment with score {segment['score']:.2f}")
            break
    else:
        print(f"Warning: {target_label} not found in segmentation")
    
    # Save mask
    mask_image = Image.fromarray(mask, mode='L')
    mask_image.save(output_path)
    print(f"Mask saved to {output_path}")

if __name__ == "__main__":
    # Create mask for the first test image
    create_mask_from_segmentation(
        "/Users/nav/style-up/test_images/girl.jpg_mask_data.json",
        "/Users/nav/style-up/test_images/girl_upper_clothes_mask.png",
        "Upper-clothes"
    )