#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Model Testing Script

This script tests the trained DestripeNet model on a small batch of images
and visualizes the results.
"""

import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from destripenet import DestripeNet
from train import LunarDataset
from torch.utils.data import DataLoader

def test_model(model, test_loader, device, num_samples=5, save_dir='../results'):
    """
    Test the model on a few samples and visualize the results.
    
    Args:
        model: The trained DestripeNet model
        test_loader: DataLoader for test data
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_dir: Directory to save results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Get a batch of samples
    noisy_imgs, clean_imgs = next(iter(test_loader))
    
    # Limit to num_samples
    noisy_imgs = noisy_imgs[:num_samples]
    clean_imgs = clean_imgs[:num_samples]
    
    # Move to device
    noisy_imgs = noisy_imgs.to(device)
    clean_imgs = clean_imgs.to(device)
    
    # Forward pass
    with torch.no_grad():
        enhanced_imgs = model(noisy_imgs)
    
    # Create figure for visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Get images
        noisy_img = noisy_imgs[i].cpu().numpy().squeeze()
        clean_img = clean_imgs[i].cpu().numpy().squeeze()
        enhanced_img = enhanced_imgs[i].cpu().numpy().squeeze()
        
        # Plot images
        axes[i, 0].imshow(noisy_img, cmap='gray')
        axes[i, 0].set_title('Noisy (Input)')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(enhanced_img, cmap='gray')
        axes[i, 1].set_title('Enhanced (Output)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(clean_img, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_results.png'))
    plt.close()
    
    print(f"Test results saved to {os.path.join(save_dir, 'test_results.png')}")
    
    # Save individual images for closer inspection
    for i in range(num_samples):
        noisy_img = (noisy_imgs[i].cpu().numpy().squeeze() * 255).astype(np.uint8)
        clean_img = (clean_imgs[i].cpu().numpy().squeeze() * 255).astype(np.uint8)
        enhanced_img = (enhanced_imgs[i].cpu().numpy().squeeze() * 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(save_dir, f'sample_{i}_noisy.png'), noisy_img)
        cv2.imwrite(os.path.join(save_dir, f'sample_{i}_clean.png'), clean_img)
        cv2.imwrite(os.path.join(save_dir, f'sample_{i}_enhanced.png'), enhanced_img)

def main():
    parser = argparse.ArgumentParser(description="Test DestripeNet for lunar image enhancement")
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing the dataset')
    parser.add_argument('--model_path', type=str, default='../models/best_model.pth', help='Path to the trained model')
    parser.add_argument('--save_dir', type=str, default='../results', help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on (cuda or cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create test dataset and dataloader
    test_dataset = LunarDataset(os.path.join(args.data_dir, 'test'))
    test_loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=True)
    
    # Create model and load weights
    model = DestripeNet(in_channels=1, out_channels=1).to(device)
    
    # Load model weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model for testing.")
    
    # Test the model
    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
    
    print("Testing complete!")

if __name__ == "__main__":
    main()
