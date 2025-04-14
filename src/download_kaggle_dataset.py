#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Kaggle Dataset Downloader

This script downloads the Artificial Lunar Landscape Dataset from Kaggle
and prepares it for training the DestripeNet model.
"""

import os
import sys
import subprocess
import zipfile
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import random
import argparse

def setup_kaggle_credentials():
    """Set up Kaggle API credentials."""
    # Create Kaggle directory if it doesn't exist
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    
    # Create a kaggle.json file with API credentials
    # Note: In a real scenario, you would need to get these from the user
    # For this demo, we'll create a placeholder and inform the user
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    
    if not os.path.exists(kaggle_json_path):
        print("Kaggle API credentials are required to download the dataset.")
        print("Please create a kaggle.json file with your API credentials.")
        print("You can get your API credentials from https://www.kaggle.com/account")
        print("Creating a placeholder kaggle.json file...")
        
        # Create a placeholder file
        with open(kaggle_json_path, 'w') as f:
            f.write('{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}')
        
        # Set proper permissions
        os.chmod(kaggle_json_path, 0o600)
        
        print(f"Placeholder file created at {kaggle_json_path}")
        print("Please update this file with your actual Kaggle credentials.")
        return False
    
    return True

def download_kaggle_dataset():
    """Download the Artificial Lunar Landscape Dataset from Kaggle."""
    dataset_name = "romainpessia/artificial-lunar-rocky-landscape-dataset"
    download_path = "../data/raw"
    
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    print(f"Downloading dataset {dataset_name} to {download_path}...")
    
    try:
        # Use kaggle CLI to download the dataset
        subprocess.run(
            ["kaggle", "datasets", "download", dataset_name, "--path", download_path],
            check=True
        )
        
        # Extract the downloaded zip file
        zip_file = os.path.join(download_path, "artificial-lunar-rocky-landscape-dataset.zip")
        if os.path.exists(zip_file):
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            
            # Remove the zip file after extraction
            os.remove(zip_file)
            print("Dataset downloaded and extracted successfully.")
            return True
        else:
            print(f"Downloaded zip file not found at {zip_file}")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def create_directories():
    """Create necessary directories for the dataset."""
    os.makedirs("../data/raw", exist_ok=True)
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../data/train", exist_ok=True)
    os.makedirs("../data/val", exist_ok=True)
    os.makedirs("../data/test", exist_ok=True)
    print("Created dataset directories")

def prepare_dataset():
    """
    Prepare the downloaded dataset for training.
    
    This function:
    1. Organizes the images into train/val/test splits
    2. Creates pairs of (dark, normal) images for training
    3. Processes images to simulate dark regions on the moon
    """
    raw_dir = "../data/raw"
    train_dir = "../data/train"
    val_dir = "../data/val"
    test_dir = "../data/test"
    
    # Check if the dataset was downloaded and extracted
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0:
        print("Raw data directory is empty. Please download the dataset first.")
        return False
    
    print("Preparing dataset for training...")
    
    # Find all image files in the raw directory
    image_files = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print("No image files found in the raw data directory.")
        return False
    
    print(f"Found {len(image_files)} image files.")
    
    # Shuffle and split the dataset
    random.shuffle(image_files)
    train_split = int(0.8 * len(image_files))
    val_split = int(0.9 * len(image_files))
    
    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]
    
    print(f"Split dataset: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test images.")
    
    # Process images for each split
    process_images(train_files, train_dir, "train")
    process_images(val_files, val_dir, "val")
    process_images(test_files, test_dir, "test")
    
    print("Dataset preparation complete.")
    return True

def process_images(image_files, output_dir, split_name):
    """
    Process images to create pairs of (dark, normal) images.
    
    Args:
        image_files: List of image file paths
        output_dir: Directory to save processed images
        split_name: Name of the split (train, val, test)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(image_files)} images for {split_name} split...")
    
    for i, img_path in enumerate(tqdm(image_files)):
        try:
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
            cv2.imwrite(f"{output_dir}/noisy_{i:04d}.png", noisy)
            cv2.imwrite(f"{output_dir}/clean_{i:04d}.png", clean)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    print(f"Processed {len(image_files)} images for {split_name} split.")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare lunar image dataset from Kaggle")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading the dataset")
    parser.add_argument("--skip-processing", action="store_true", help="Skip processing the dataset")
    args = parser.parse_args()
    
    create_directories()
    
    if not args.skip_download:
        if setup_kaggle_credentials():
            download_kaggle_dataset()
        else:
            print("Kaggle credentials setup incomplete. Skipping download.")
    
    if not args.skip_processing:
        prepare_dataset()
    
    print("Dataset preparation complete")

if __name__ == "__main__":
    main()
