# Lunar Image Enhancement Project Documentation

## Overview

This project focuses on enhancing images of dark regions on the moon using image processing and deep learning techniques. The implementation includes a lightweight model based on classical image processing methods and a user-friendly Streamlit frontend for easy interaction.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
   - [Local Environment (VSCode)](#local-environment-vscode)
   - [Google Colab](#google-colab)
4. [Model Architecture](#model-architecture)
5. [Dataset](#dataset)
6. [Training Process](#training-process)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Frontend Interface](#frontend-interface)
9. [Future Improvements](#future-improvements)

## Project Structure

```
lunar_image_enhancement/
├── data/                  # Dataset directory
│   ├── train/             # Training images
│   ├── val/               # Validation images
│   └── test/              # Test images
├── src/                   # Source code
│   ├── destripenet.py     # DestripeNet model implementation
│   ├── lightweight_model.py # Lightweight model implementation
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   ├── test.py            # Testing script
│   └── inference.py       # Inference script
├── models/                # Saved model weights
├── results/               # Results and visualizations
├── frontend/              # Streamlit frontend
│   ├── app.py             # Streamlit application
│   └── assets/            # Frontend assets
└── docs/                  # Documentation
    └── README.md          # This documentation
```

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package installer)

### Dependencies

The project requires the following main dependencies:

```
numpy
opencv-python
matplotlib
scipy
streamlit
```

For the full DestripeNet implementation (not included in the lightweight version), you would also need:

```
torch
torchvision
```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lunar_image_enhancement.git
   cd lunar_image_enhancement
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Local Environment (VSCode)

1. Open the project in VSCode:
   ```bash
   code .
   ```

2. Run the Streamlit application:
   ```bash
   cd lunar_image_enhancement
   streamlit run frontend/app.py
   ```

3. The application will open in your default web browser at `http://localhost:8501`.

### Google Colab

1. Upload the project to Google Drive.

2. Create a new Colab notebook and mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Navigate to the project directory:
   ```python
   %cd /content/drive/MyDrive/path/to/lunar_image_enhancement
   ```

4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```

5. Run the Streamlit application with localtunnel:
   ```python
   !pip install streamlit pyngrok
   from pyngrok import ngrok
   
   # Start Streamlit in the background
   !streamlit run frontend/app.py &
   
   # Create a tunnel to the Streamlit app
   public_url = ngrok.connect(8501)
   print(f"Public URL: {public_url}")
   ```

6. Click on the public URL to access the application.

## Model Architecture

### DestripeNet Architecture

The DestripeNet model is designed for enhancing dark regions in lunar images. It is based on a U-Net architecture with residual blocks and attention mechanisms. The model consists of:

1. **Encoder**: Extracts features from the input image through convolutional layers and downsampling.
2. **Bottleneck**: Processes the encoded features with residual blocks and attention mechanisms.
3. **Decoder**: Reconstructs the enhanced image through upsampling and convolutional layers.
4. **Skip Connections**: Connect encoder and decoder layers to preserve spatial information.

### Lightweight Model

Due to environment constraints, a lightweight model was implemented using classical image processing techniques:

1. **Adaptive Histogram Equalization (CLAHE)**: Improves contrast in local regions.
2. **Bilateral Filtering**: Reduces noise while preserving edges.
3. **Unsharp Masking**: Enhances details by subtracting a blurred version from the original.
4. **Gamma Correction**: Adjusts brightness in dark regions while preserving details.

## Dataset

The project uses a synthetic dataset of lunar images with simulated dark regions. The dataset consists of:

- 160 training images
- 20 validation images
- 20 test images

Each sample contains a pair of images:
- Clean lunar surface image (ground truth)
- Corresponding darkened/noisy version (input)

## Training Process

### Full DestripeNet Training

The training process for the full DestripeNet model involves:

1. **Data Preparation**: Loading and preprocessing the dataset.
2. **Model Initialization**: Creating the DestripeNet model with appropriate parameters.
3. **Loss Function**: Using a combination of L1 loss and SSIM (Structural Similarity) loss.
4. **Optimization**: Using Adam optimizer with learning rate scheduling.
5. **Training Loop**: Training for multiple epochs with validation.
6. **Checkpointing**: Saving the best model based on validation loss.

### Training Script

The training script (`train.py`) includes:
- Dataset loading and preprocessing
- Model creation and initialization
- Training loop with validation
- Metrics tracking and visualization
- Model checkpointing

## Evaluation Metrics

The model is evaluated using the following metrics:

1. **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of reconstruction.
2. **SSIM (Structural Similarity Index)**: Measures the perceived quality of the enhanced image.
3. **Visual Comparison**: Side-by-side comparison of original and enhanced images.

## Frontend Interface

The Streamlit frontend provides a user-friendly interface for interacting with the model:

1. **Image Upload**: Users can upload their own lunar images.
2. **Sample Images**: Pre-loaded sample images are available for testing.
3. **Enhancement Visualization**: Side-by-side comparison of original and enhanced images.
4. **Histogram Comparison**: Visualization of pixel value distribution before and after enhancement.
5. **Download Option**: Users can download the enhanced images.

## Future Improvements

1. **Full DestripeNet Implementation**: Implement the complete deep learning model when resources allow.
2. **Transfer Learning**: Fine-tune the model on real lunar images from NASA or other sources.
3. **Multi-Scale Processing**: Implement multi-scale processing for better handling of different image resolutions.
4. **Batch Processing**: Add functionality to process multiple images at once.
5. **Advanced Visualization**: Add more visualization options like zooming into specific regions.

## Running in Different Environments

### VSCode

1. Open the project in VSCode.
2. Install the Python extension if not already installed.
3. Select your Python interpreter (preferably from a virtual environment).
4. Open a terminal in VSCode and run:
   ```bash
   streamlit run frontend/app.py
   ```

### Google Colab

Create a new notebook with the following cells:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository (if not already in Drive)
!git clone https://github.com/yourusername/lunar_image_enhancement.git
%cd lunar_image_enhancement

# Install dependencies
!pip install -r requirements.txt

# Run the lightweight model demo
!python src/lightweight_model.py

# Set up Streamlit with ngrok
!pip install streamlit pyngrok
from pyngrok import ngrok

# Start Streamlit
!streamlit run frontend/app.py &
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")
```

Click on the generated public URL to access the application.
