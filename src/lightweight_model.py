#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Lightweight Model Implementation

This script provides a lightweight implementation of the DestripeNet model
that can be used for demonstration purposes without requiring PyTorch.
"""

import numpy as np
import cv2
from scipy import ndimage

class LightweightDestripeNet:
    """
    A lightweight implementation of DestripeNet for enhancing dark regions on the moon.
    
    This class uses classical image processing techniques to simulate the effects
    of the deep learning model without requiring PyTorch.
    """
    
    def __init__(self):
        """Initialize the lightweight model."""
        pass
    
    def enhance_image(self, image):
        """
        Enhance a dark lunar image.
        
        Args:
            image: Input image (numpy array, grayscale)
            
        Returns:
            Enhanced image
        """
        # Ensure image is grayscale and in the range [0, 255]
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert to float for processing
        image_float = image.astype(np.float32) / 255.0
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(np.uint8(image_float * 255)) / 255.0
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(np.uint8(clahe_image * 255), 9, 75, 75) / 255.0
        
        # Apply unsharp masking to enhance details
        gaussian = ndimage.gaussian_filter(bilateral, sigma=1.0)
        unsharp_image = bilateral + 0.5 * (bilateral - gaussian)
        unsharp_image = np.clip(unsharp_image, 0.0, 1.0)
        
        # Enhance contrast in dark regions
        gamma = 0.7
        gamma_corrected = np.power(unsharp_image, gamma)
        
        # Final result
        result = np.clip(gamma_corrected, 0.0, 1.0)
        
        return (result * 255).astype(np.uint8)
    
    def process_batch(self, images):
        """
        Process a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of enhanced images
        """
        return [self.enhance_image(img) for img in images]

def demo():
    """
    Demonstrate the lightweight model on a sample image.
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create a sample dark lunar image
    def create_sample_image(size=512):
        # Create base surface with noise
        surface = np.random.normal(128, 20, (size, size)).astype(np.uint8)
        
        # Apply Gaussian blur to make it smoother
        surface = cv2.GaussianBlur(surface, (15, 15), 0)
        
        # Add craters
        for _ in range(50):
            # Random crater parameters
            center_x = np.random.randint(0, size-1)
            center_y = np.random.randint(0, size-1)
            radius = np.random.randint(5, 50)
            rim_brightness = np.random.randint(10, 40)
            center_darkness = np.random.randint(10, 30)
            
            # Create crater
            y, x = np.ogrid[:size, :size]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Crater rim (brighter)
            rim_mask = (dist_from_center >= radius*0.8) & (dist_from_center <= radius*1.2)
            surface[rim_mask] = np.clip(surface[rim_mask] + rim_brightness, 0, 255)
            
            # Crater center (darker)
            center_mask = dist_from_center < radius*0.8
            surface[center_mask] = np.clip(surface[center_mask] - center_darkness, 0, 255)
        
        # Create a dark region
        mask = np.zeros_like(surface)
        points = np.random.randint(0, size, (5, 2))
        cv2.fillPoly(mask, [points], 255)
        
        # Apply darkening
        darkness = 0.3
        dark_image = surface.copy()
        dark_image[mask > 0] = dark_image[mask > 0] * darkness
        
        # Add noise
        noise = np.random.normal(0, 15, dark_image.shape).astype(np.int16)
        noisy_image = np.clip(dark_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    # Create sample image
    sample_image = create_sample_image()
    
    # Create model and enhance image
    model = LightweightDestripeNet()
    enhanced_image = model.enhance_image(sample_image)
    
    # Create results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)
    
    # Save images
    cv2.imwrite("../results/sample_dark_region.png", sample_image)
    cv2.imwrite("../results/sample_enhanced.png", enhanced_image)
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image, cmap='gray')
    plt.title('Dark Lunar Region (Input)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('Enhanced Image (Output)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("../results/enhancement_demo.png")
    
    print("Demo completed. Results saved to '../results/' directory.")

if __name__ == "__main__":
    demo()
