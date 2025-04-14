#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Evaluation Script

This script evaluates the trained DestripeNet model on the test dataset.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from destripenet import DestripeNet
from train import LunarDataset

def evaluate(model, test_loader, device, save_dir):
    """
    Evaluate the DestripeNet model on the test dataset.
    
    Args:
        model: The trained DestripeNet model
        test_loader: DataLoader for test data
        device: Device to evaluate on (cuda or cpu)
        save_dir: Directory to save evaluation results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    
    model.eval()
    
    # Lists to store metrics
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for i, (noisy_imgs, clean_imgs) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move data to device
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            
            # Convert to numpy for evaluation and visualization
            for j in range(noisy_imgs.size(0)):
                # Get individual images
                noisy_img = noisy_imgs[j].cpu().numpy().squeeze()
                clean_img = clean_imgs[j].cpu().numpy().squeeze()
                output_img = outputs[j].cpu().numpy().squeeze()
                
                # Calculate metrics
                psnr_value = psnr(clean_img, output_img, data_range=1.0)
                ssim_value = ssim(clean_img, output_img, data_range=1.0)
                
                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                
                # Save some example results
                if i < 10:  # Save first 10 batches
                    # Create comparison figure
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(noisy_img, cmap='gray')
                    axes[0].set_title(f'Noisy (Input)')
                    axes[0].axis('off')
                    
                    axes[1].imshow(output_img, cmap='gray')
                    axes[1].set_title(f'Enhanced (Output)\nPSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}')
                    axes[1].axis('off')
                    
                    axes[2].imshow(clean_img, cmap='gray')
                    axes[2].set_title('Ground Truth')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'results', f'result_{i}_{j}.png'))
                    plt.close()
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Save metrics to file
    with open(os.path.join(save_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Number of test images: {len(psnr_values)}\n")
    
    # Plot histogram of metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(psnr_values, bins=20, alpha=0.7)
    plt.axvline(avg_psnr, color='r', linestyle='dashed', linewidth=2)
    plt.title(f'PSNR Distribution (Avg: {avg_psnr:.2f} dB)')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(ssim_values, bins=20, alpha=0.7)
    plt.axvline(avg_ssim, color='r', linestyle='dashed', linewidth=2)
    plt.title(f'SSIM Distribution (Avg: {avg_ssim:.4f})')
    plt.xlabel('SSIM')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'))
    
    return avg_psnr, avg_ssim

def main():
    parser = argparse.ArgumentParser(description="Evaluate DestripeNet for lunar image enhancement")
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing the dataset')
    parser.add_argument('--model_path', type=str, default='../models/best_model.pth', help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='../evaluation', help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to evaluate on (cuda or cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create test dataset and dataloader
    test_dataset = LunarDataset(os.path.join(args.data_dir, 'test'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model and load weights
    model = DestripeNet(in_channels=1, out_channels=1).to(device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate the model
    avg_psnr, avg_ssim = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir
    )
    
    print("Evaluation complete!")
    print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
