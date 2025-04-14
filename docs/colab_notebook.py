#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Google Colab Notebook

This notebook demonstrates how to run the lunar image enhancement project in Google Colab.
"""

# @title # Lunar Image Enhancement Project
# @markdown This notebook allows you to run the lunar image enhancement project in Google Colab.

# @markdown ## Setup
# @markdown Run this cell to set up the environment and download the project.

# Install required packages
!pip install -q numpy opencv-python matplotlib scipy streamlit pillow pyngrok

# Clone the repository
!git clone https://github.com/yourusername/lunar_image_enhancement.git
%cd lunar_image_enhancement

# Install project requirements
!pip install -q -r requirements.txt

# @markdown ## Run the Demo
# @markdown Run this cell to test the lightweight model on a sample image.

# Run the lightweight model demo
!python src/lightweight_model.py

# Display the results
from IPython.display import Image, display
import os

if os.path.exists("results/enhancement_demo.png"):
    display(Image("results/enhancement_demo.png"))
    print("Demo completed successfully!")
else:
    print("Demo image not found. Please check for errors above.")

# @markdown ## Launch Streamlit App
# @markdown Run this cell to launch the Streamlit app with a public URL.

# Set up and run Streamlit with ngrok
from pyngrok import ngrok
import subprocess
import time

# Start Streamlit in the background
process = subprocess.Popen(["streamlit", "run", "frontend/app.py"], stdout=subprocess.PIPE)

# Wait for Streamlit to start
time.sleep(5)

# Create a tunnel to the Streamlit app
public_url = ngrok.connect(8501)
print(f"Streamlit app is running at: {public_url}")
print("Click the link above to access the application.")
print("Note: The link will be active for 2 hours or until this notebook session ends.")

# @markdown ## Upload Your Own Images
# @markdown You can upload your own lunar images to test the enhancement.

# Code for uploading and processing custom images
from google.colab import files
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.lightweight_model import LightweightDestripeNet

def process_uploaded_image():
    print("Please upload a lunar image...")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        # Read the image
        file_bytes = uploaded[filename]
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Initialize the model
        model = LightweightDestripeNet()
        
        # Process the image
        enhanced_image = model.enhance_image(image)
        
        # Display the results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_image, cmap='gray')
        plt.title('Enhanced Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the enhanced image
        output_filename = f"enhanced_{filename}"
        cv2.imwrite(output_filename, enhanced_image)
        files.download(output_filename)
        
        print(f"Enhanced image saved as {output_filename} and downloaded.")

# Run the function when this cell is executed
process_uploaded_image()

# @markdown ## Cleanup
# @markdown Run this cell to clean up resources when you're done.

# Disconnect ngrok tunnels
ngrok.kill()

# Terminate the Streamlit process
process.terminate()

print("Resources cleaned up successfully!")
