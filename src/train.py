#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Training Script

This script trains the DestripeNet model for enhancing dark regions on the moon.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from destripenet import DestripeNet, DestripeNetLoss

class LunarDataset(Dataset):
    """
    Dataset class for lunar images.
    
    Args:
        data_dir: Directory containing the image pairs
        transform: Optional transforms to apply to the images
        split: 'train', 'val', or 'test'
    """
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Get all noisy and clean image pairs
        self.noisy_images = sorted([f for f in os.listdir(data_dir) if f.startswith('noisy_')])
        self.clean_images = sorted([f for f in os.listdir(data_dir) if f.startswith('clean_')])
        
        # Ensure we have matching pairs
        assert len(self.noisy_images) == len(self.clean_images), "Mismatch in number of noisy and clean images"
    
    def __len__(self):
        return len(self.noisy_images)
    
    def __getitem__(self, idx):
        # Load images
        noisy_path = os.path.join(self.data_dir, self.noisy_images[idx])
        clean_path = os.path.join(self.data_dir, self.clean_images[idx])
        
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize to [0, 1]
        noisy_img = noisy_img.astype(np.float32) / 255.0
        clean_img = clean_img.astype(np.float32) / 255.0
        
        # Convert to tensors
        noisy_img = torch.from_numpy(noisy_img).unsqueeze(0)  # Add channel dimension
        clean_img = torch.from_numpy(clean_img).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if specified
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_dir):
    """
    Train the DestripeNet model.
    
    Args:
        model: The DestripeNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda or cpu)
        num_epochs: Number of epochs to train for
        save_dir: Directory to save model checkpoints
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for noisy_imgs, clean_imgs in pbar:
                # Move data to device
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                
                # Forward pass
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                for noisy_imgs, clean_imgs in pbar:
                    # Move data to device
                    noisy_imgs = noisy_imgs.to(device)
                    clean_imgs = clean_imgs.to(device)
                    
                    # Forward pass
                    outputs = model(noisy_imgs)
                    loss = criterion(outputs, clean_imgs)
                    
                    # Update metrics
                    val_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot and save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, os.path.join(save_dir, 'final_model.pth'))
    
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description="Train DestripeNet for lunar image enhancement")
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='../models', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to train on (cuda or cpu)')
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = LunarDataset(os.path.join(args.data_dir, 'train'))
    val_dataset = LunarDataset(os.path.join(args.data_dir, 'val'))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model, loss function, and optimizer
    model = DestripeNet(in_channels=1, out_channels=1).to(device)
    criterion = DestripeNetLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
