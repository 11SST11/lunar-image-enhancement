import streamlit as st
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import io

# Add the src directory to the path so we can import the lightweight model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.lightweight_model import LightweightDestripeNet, demo

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Lunar Image Enhancement",
        page_icon="🌒",
        layout="wide"
    )
    
    # Create sidebar
    st.sidebar.title("Lunar Image Enhancement")
    
    # Check if the demo image exists, if not create it
    demo_image_path = os.path.join(parent_dir, "results", "enhancement_demo.png")
    if not os.path.exists(demo_image_path):
        # Create the results directory if it doesn't exist
        os.makedirs(os.path.dirname(demo_image_path), exist_ok=True)
        # Create a demo image
        demo()
    
    # Now display the demo image
    if os.path.exists(demo_image_path):
        st.sidebar.image(demo_image_path, use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ## About
    This application enhances images of dark regions on the moon using 
    image processing techniques inspired by DestripeNet architecture.
    
    Upload your lunar images to see the enhancement in action!
    """)
    
    # Main content
    st.title("🌒 Lunar Dark Region Enhancement")
    st.markdown("""
    This tool enhances visibility in dark regions of lunar images, revealing details that are 
    normally hidden in the shadows. Upload your lunar image below to see the enhancement in action.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a lunar image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Create model instance
        model = LightweightDestripeNet()
        
        # Process the image
        enhanced_image = model.enhance_image(image_np)
        
        # Display original and enhanced images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)
        
        # Add download button for enhanced image
        enhanced_pil = Image.fromarray(enhanced_image)
        buf = io.BytesIO()
        enhanced_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Download Enhanced Image",
            data=byte_im,
            file_name="enhanced_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
