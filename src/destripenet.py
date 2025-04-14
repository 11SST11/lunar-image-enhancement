import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual
        return self.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv_spatial = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Spatial attention
        spatial_attn = self.sigmoid(self.conv_spatial(x))
        return x * spatial_attn

class DestripeNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(DestripeNet, self).__init__()
        
        # Initial feature extraction
        self.conv_in = ConvBlock(in_channels, features)
        
        # Encoder blocks
        self.enc1 = nn.Sequential(
            ConvBlock(features, features),
            ResidualBlock(features)
        )
        self.down1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            ConvBlock(features, features*2),
            ResidualBlock(features*2)
        )
        self.down2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            ConvBlock(features*2, features*4),
            ResidualBlock(features*4)
        )
        self.down3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(features*4, features*8),
            ResidualBlock(features*8),
            AttentionBlock(features*8),
            ConvBlock(features*8, features*4)
        )
        
        # Decoder blocks
        self.up3 = nn.ConvTranspose2d(features*4, features*4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBlock(features*8, features*4),
            ResidualBlock(features*4)
        )
        
        self.up2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(features*4, features*2),
            ResidualBlock(features*2)
        )
        
        self.up1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(features*2, features),
            ResidualBlock(features),
            AttentionBlock(features)
        )
        
        # Output layer
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Initial feature extraction
        x1 = self.conv_in(x)
        
        # Encoder
        e1 = self.enc1(x1)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.down3(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Output
        out = self.conv_out(d1)
        
        return out

class DestripeNetLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.5):
        super(DestripeNetLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        
    def forward(self, pred, target):
        l1_loss = self.l1_loss(pred, target)
        
        # SSIM loss component (structural similarity)
        ssim_loss = 1 - self.ssim(pred, target)
        
        # Combined loss
        total_loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss
        
        return total_loss
    
    def ssim(self, img1, img2, window_size=11, size_average=True):
        # SSIM implementation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create window
        window = self.create_window(window_size, img1.size(1))
        window = window.to(img1.device)
        
        # Calculate means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
