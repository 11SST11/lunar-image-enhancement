#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Synthetic Dataset Generator

This script generates a synthetic dataset for training the DestripeNet model
for enhancing dark regions on the moon, without requiring Kaggle credentials.
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import random
import argparse

def create_directories():
    """Create necessary directories for the dataset."""
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../data/train", exist_ok=True)
    os.makedirs("../data/val", exist_ok=True)
    os.makedirs("../data/test", exist_ok=True)
    print("Created dataset directories")

def generate_synthetic_moon_surface(size=512, num_craters=50, num_rocks=100):
    """
    Generate a synthetic moon surface image with craters and rocks.
    
    Args:
        size: Size of the square image
        num_craters: Number of craters to generate
        num_rocks: Number of rocks to generate
        
    Returns:
        Synthetic moon surface image
    """
    # Create base surface with noise
    surface = np.random.normal(128, 20, (size, size)).astype(np.uint8)
    
    # Apply Gaussian blur to make it smoother
    surface = cv2.GaussianBlur(surface, (15, 15), 0)
    
    # Add large-scale terrain variations
    for _ in range(5):
        center_x = random.randint(0, size)
        center_y = random.randint(0, size)
        radius = random.randint(size//4, size//2)
        intensity = random.randint(-30, 30)
        
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist_from_center <= radius
        
        # Create a gradient effect
        gradient = np.clip(1 - (dist_from_center / radius), 0, 1)
        gradient = gradient * intensity
        
        # Apply the terrain variation
        surface = np.clip(surface + gradient * mask, 0, 255).astype(np.uint8)
    
    # Add craters
    for _ in range(num_craters):
        # Random crater parameters
        center_x = random.randint(0, size-1)
        center_y = random.randint(0, size-1)
        radius = random.randint(5, 50)
        rim_brightness = random.randint(10, 40)
        center_darkness = random.randint(10, 30)
        
        # Create crater
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Crater rim (brighter)
        rim_mask = (dist_from_center >= radius*0.8) & (dist_from_center <= radius*1.2)
        surface[rim_mask] = np.clip(surface[rim_mask] + rim_brightness, 0, 255)
        
        # Crater center (darker)
        center_mask = dist_from_center < radius*0.8
        surface[center_mask] = np.clip(surface[center_mask] - center_darkness, 0, 255)
    
    # Add rocks
    for _ in range(num_rocks):
        # Random rock parameters
        center_x = random.randint(0, size-1)
        center_y = random.randint(0, size-1)
        radius = random.randint(1, 8)
        brightness = random.randint(-20, 20)
        
        # Create rock
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        rock_mask = dist_from_center <= radius
        
        # Apply rock - convert to int16 before adding to avoid overflow
        surface_int16 = surface[rock_mask].astype(np.int16)
        surface[rock_mask] = np.clip(surface_int16 + brightness, 0, 255).astype(np.uint8)
    
    return surface

def add_shadows_and_noise(image, shadow_intensity=0.3, noise_level=30):
    """
    Add shadows and noise to simulate dark regions on the moon.
    
    Args:
        image: Input image
        shadow_intensity: Intensity of shadows (lower means darker)
        noise_level: Level of noise to add
        
    Returns:
        Image with shadows and noise
    """
    # Create a copy of the image
    shadowed = image.copy()
    
    # Create random shadow shapes
    mask = np.zeros_like(image)
    
    # Number of shadow regions
    num_shadows = random.randint(1, 3)
    
    for _ in range(num_shadows):
        # Create a random polygon for shadow
        num_points = random.randint(3, 7)
        points = np.random.randint(0, image.shape[0], (num_points, 2))
        cv2.fillPoly(mask, [points], 255)
    
    # Apply darkening to shadowed regions
    shadowed[mask > 0] = shadowed[mask > 0] * shadow_intensity
    
    # Add noise to the entire image, with more in shadowed regions
    base_noise = np.random.normal(0, noise_level/2, image.shape).astype(np.int16)
    shadow_noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    
    # Apply base noise to entire image
    noisy = np.clip(shadowed.astype(np.int16) + base_noise, 0, 255).astype(np.uint8)
    
    # Apply extra noise to shadowed regions
    noisy[mask > 0] = np.clip(noisy[mask > 0].astype(np.int16) + shadow_noise[mask > 0], 0, 255).astype(np.uint8)
    
    return noisy, image  # Return both noisy and clean versions

def generate_dataset(num_samples=500, image_size=512):
    """
    Generate a synthetic dataset for training.
    
    Args:
        num_samples: Number of image pairs to generate
        image_size: Size of the images
    """
    print(f"Generating {num_samples} synthetic image pairs...")
    
    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    # test_ratio = 0.1 (remainder)
    
    train_samples = int(num_samples * train_ratio)
    val_samples = int(num_samples * val_ratio)
    test_samples = num_samples - train_samples - val_samples
    
    print(f"Split: {train_samples} training, {val_samples} validation, {test_samples} test samples")
    
    # Generate samples
    for i in tqdm(range(num_samples)):
        # Determine which set this sample belongs to
        if i < train_samples:
            output_dir = "../data/train"
        elif i < train_samples + val_samples:
            output_dir = "../data/val"
        else:
            output_dir = "../data/test"
        
        # Generate synthetic moon surface
        num_craters = random.randint(20, 100)
        num_rocks = random.randint(50, 200)
        moon_surface = generate_synthetic_moon_surface(image_size, num_craters, num_rocks)
        
        # Add shadows and noise
        shadow_intensity = random.uniform(0.1, 0.5)
        noise_level = random.uniform(10, 50)
        noisy_image, clean_image = add_shadows_and_noise(moon_surface, shadow_intensity, noise_level)
        
        # Save the image pair
        cv2.imwrite(f"{output_dir}/noisy_{i:04d}.png", noisy_image)
        cv2.imwrite(f"{output_dir}/clean_{i:04d}.png", clean_image)
    
    print(f"Generated {num_samples} synthetic image pairs")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic lunar image dataset")
    parser.add_argument("--samples", type=int, default=500, help="Number of image pairs to generate")
    parser.add_argument("--size", type=int, default=512, help="Size of the images")
    args = parser.parse_args()
    
    create_directories()
    generate_dataset(args.samples, args.size)
    
    print("Dataset generation complete")

if __name__ == "__main__":
    main()
