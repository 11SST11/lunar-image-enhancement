#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Dataset Downloader

This script downloads a sample of lunar images from the LROC dataset,
focusing on dark regions and permanently shadowed regions (PSRs).
"""

import os
import requests
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import random
import time
import argparse

# Base URLs for LROC data
LROC_BASE_URL = "https://wms.lroc.asu.edu/lroc/view_lroc"
LROC_SEARCH_URL = "https://wms.lroc.asu.edu/lroc/search"

# Keywords for finding dark region images
DARK_REGION_KEYWORDS = [
    "permanently shadowed region",
    "PSR",
    "dark region",
    "shadow",
    "lunar night",
    "polar crater"
]

def create_directories():
    """Create necessary directories for the dataset."""
    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../data/train", exist_ok=True)
    os.makedirs("../data/val", exist_ok=True)
    os.makedirs("../data/test", exist_ok=True)
    print("Created dataset directories")

def download_sample_images():
    """
    Download sample images from LROC.
    
    Since we can't directly access the LROC database API, we'll download
    a set of sample images that are publicly available.
    """
    # Sample image URLs (these would normally come from an API search)
    # These are example URLs - in a real scenario, we would query the LROC database
    sample_urls = [
        "https://wms.lroc.asu.edu/lroc/browse_image/nac/M1096293859LE",
        "https://wms.lroc.asu.edu/lroc/browse_image/nac/M1096293859RE",
        "https://wms.lroc.asu.edu/lroc/browse_image/nac/M1108074212LE",
        "https://wms.lroc.asu.edu/lroc/browse_image/nac/M1108074212RE",
        "https://wms.lroc.asu.edu/lroc/browse_image/nac/M1142582036LE",
        "https://wms.lroc.asu.edu/lroc/browse_image/nac/M1142582036RE",
    ]
    
    print(f"Downloading {len(sample_urls)} sample images...")
    
    for i, url in enumerate(sample_urls):
        try:
            # Extract image ID from URL
            image_id = url.split('/')[-1]
            output_path = f"../data/raw/{image_id}.jpg"
            
            # Skip if already downloaded
            if os.path.exists(output_path):
                print(f"Image {image_id} already exists, skipping...")
                continue
                
            # Download the image
            print(f"Downloading image {i+1}/{len(sample_urls)}: {image_id}")
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                print(f"Downloaded {image_id}")
            else:
                print(f"Failed to download {image_id}: HTTP {response.status_code}")
                
            # Be nice to the server
            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    print("Download complete")

def generate_synthetic_dataset():
    """
    Generate a synthetic dataset for training.
    
    Since we may not have access to the full LROC dataset, we'll create
    a synthetic dataset by:
    1. Taking normal lunar images
    2. Darkening parts of them to simulate shadowed regions
    3. Adding noise to simulate low-light conditions
    4. Creating pairs of (noisy, clean) images for training
    """
    print("Generating synthetic dataset for training...")
    
    # Get list of downloaded images
    raw_images = [f for f in os.listdir("../data/raw") if f.endswith(('.jpg', '.png'))]
    
    if not raw_images:
        print("No raw images found. Please download images first.")
        return
    
    # Number of synthetic pairs to generate
    num_samples = 100
    
    for i in tqdm(range(num_samples)):
        # Randomly select an image
        img_file = random.choice(raw_images)
        img_path = os.path.join("../data/raw", img_file)
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Resize to a standard size
        gray = cv2.resize(gray, (512, 512))
        
        # Create a clean version (ground truth)
        clean = gray.copy()
        
        # Create a darkened version (simulate shadowed region)
        # Randomly darken parts of the image
        mask = np.zeros_like(gray)
        
        # Create random shadow shapes
        for _ in range(random.randint(1, 3)):
            # Random polygon for shadow
            points = np.random.randint(0, 512, (random.randint(3, 7), 2))
            cv2.fillPoly(mask, [points], 255)
        
        # Apply darkening
        darkness = random.uniform(0.1, 0.5)  # Random darkness level
        dark = gray.copy()
        dark[mask > 0] = dark[mask > 0] * darkness
        
        # Add noise to simulate low-light conditions
        noise_level = random.uniform(10, 50)
        noise = np.random.normal(0, noise_level, dark.shape).astype(np.uint8)
        noisy = cv2.add(dark, noise)
        
        # Save the pairs
        set_type = "train" if i < 80 else "val" if i < 90 else "test"
        cv2.imwrite(f"../data/{set_type}/noisy_{i:04d}.png", noisy)
        cv2.imwrite(f"../data/{set_type}/clean_{i:04d}.png", clean)
    
    print(f"Generated {num_samples} synthetic image pairs")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare lunar image dataset")
    parser.add_argument("--download-only", action="store_true", help="Only download sample images without generating synthetic data")
    parser.add_argument("--synthetic-only", action="store_true", help="Only generate synthetic data without downloading")
    args = parser.parse_args()
    
    create_directories()
    
    if not args.synthetic_only:
        download_sample_images()
    
    if not args.download_only:
        generate_synthetic_dataset()
    
    print("Dataset preparation complete")

if __name__ == "__main__":
    main()
