#!/usr/bin/env python3
"""
Minimal test script to verify the editing pipeline works.
Creates a simple synthetic test case.
"""

import numpy as np
from PIL import Image
import os

# Create a simple test image (solid color)
test_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
test_image.save('test_input.jpg')
print("Created test_input.jpg")

# Create a simple mask (white square in center)
mask = np.zeros((512, 512), dtype=np.uint8)
mask[156:356, 156:356] = 255  # 200x200 white square in center
mask_image = Image.fromarray(mask)
mask_image.save('test_mask.png')
print("Created test_mask.png")

print("\nTest files created successfully!")
print("\nYou can now test the edit script with:")
print("python edit.py --original_image test_input.jpg --mask_image test_mask.png --prompt 'bright red color' --steps 10")