#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Inference Script

This script performs inference using the trained DestripeNet model on new images.
"""

import os
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from destripenet import DestripeNet

def preprocess_image(image_path, size=512):
    """
    Preprocess an image for inference.
    
    Args:
        image_path: Path to the input image
        size: Size to resize the image to
    
    Returns:
        Preprocessed image tensor
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize image
    img = cv2.resize(img, (size, size))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return img_tensor

def enhance_image(model, image_tensor, device):
    """
    Enhance an image using the DestripeNet model.
    
    Args:
        model: The trained DestripeNet model
        image_tensor: Input image tensor
        device: Device to run inference on
    
    Returns:
        Enhanced image tensor
    """
    model.eval()
    
    with torch.no_grad():
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        output = model(image_tensor)
    
    return output

def save_result(input_tensor, output_tensor, save_path):
    """
    Save the input and enhanced images side by side.
    
    Args:
        input_tensor: Input image tensor
        output_tensor: Enhanced image tensor
        save_path: Path to save the result
    """
    # Convert tensors to numpy arrays
    input_img = input_tensor.cpu().numpy().squeeze()
    output_img = output_tensor.cpu().numpy().squeeze()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input (Dark Region)')
    axes[0].axis('off')
    
    axes[1].imshow(output_img, cmap='gray')
    axes[1].set_title('Enhanced Output')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Also save individual images
    cv2.imwrite(save_path.replace('.png', '_input.png'), (input_img * 255).astype(np.uint8))
    cv2.imwrite(save_path.replace('.png', '_enhanced.png'), (output_img * 255).astype(np.uint8))

def main():
    parser = argparse.ArgumentParser(description="Inference with DestripeNet for lunar image enhancement")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='../results', help='Directory to save enhanced images')
    parser.add_argument('--model_path', type=str, default='../models/best_model.pth', help='Path to the trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on (cuda or cpu)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model and load weights
    model = DestripeNet(in_channels=1, out_channels=1).to(device)
    
    # Load model weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading without 'model_state_dict' key
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path} (alternative method)")
        except:
            print("Failed to load model. Using untrained model.")
    
    # Get list of input images
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Enhancing images"):
        try:
            # Preprocess image
            input_path = os.path.join(args.input_dir, image_file)
            input_tensor = preprocess_image(input_path)
            
            # Enhance image
            output_tensor = enhance_image(model, input_tensor, device)
            
            # Save result
            output_path = os.path.join(args.output_dir, f"enhanced_{image_file.split('.')[0]}.png")
            save_result(input_tensor, output_tensor, output_path)
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    print(f"Enhanced images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
