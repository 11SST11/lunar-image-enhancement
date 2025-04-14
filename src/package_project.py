#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
Deployment Script

This script packages the project for deployment.
"""

import os
import shutil
import argparse
import zipfile

def create_directory_structure(base_dir):
    """Create the directory structure for deployment."""
    # Create directories if they don't exist
    os.makedirs(os.path.join(base_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "frontend"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "docs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")

def copy_source_files(src_dir, dest_dir):
    """Copy source files to the deployment directory."""
    # Copy Python files from src directory
    src_files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
    for file in src_files:
        shutil.copy2(os.path.join(src_dir, file), os.path.join(dest_dir, "src"))
    
    # Copy frontend files
    frontend_src = os.path.join(src_dir, "..", "frontend")
    frontend_dest = os.path.join(dest_dir, "frontend")
    if os.path.exists(frontend_src):
        for file in os.listdir(frontend_src):
            if file.endswith(".py"):
                shutil.copy2(os.path.join(frontend_src, file), frontend_dest)
    
    # Copy documentation files
    docs_src = os.path.join(src_dir, "..", "docs")
    docs_dest = os.path.join(dest_dir, "docs")
    if os.path.exists(docs_src):
        for file in os.listdir(docs_src):
            shutil.copy2(os.path.join(docs_src, file), docs_dest)
    
    # Copy results for demo
    results_src = os.path.join(src_dir, "..", "results")
    results_dest = os.path.join(dest_dir, "results")
    if os.path.exists(results_src):
        for file in os.listdir(results_src):
            if file.endswith((".png", ".jpg", ".jpeg")):
                shutil.copy2(os.path.join(results_src, file), results_dest)
    
    # Copy root files
    root_files = ["requirements.txt", "README.md"]
    for file in root_files:
        src_file = os.path.join(src_dir, "..", file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_dir)
    
    print(f"Copied source files to {dest_dir}")

def create_zip_archive(base_dir, output_file):
    """Create a ZIP archive of the deployment directory."""
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(base_dir))
                zipf.write(file_path, arcname)
    
    print(f"Created ZIP archive: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Package the Lunar Image Enhancement project for deployment")
    parser.add_argument("--output", type=str, default="../lunar_image_enhancement_package.zip",
                        help="Output ZIP file path")
    parser.add_argument("--temp-dir", type=str, default="../deployment_temp",
                        help="Temporary directory for packaging")
    args = parser.parse_args()
    
    # Get the absolute path of the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create temporary directory for packaging
    temp_dir = os.path.abspath(args.temp_dir)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    try:
        # Create directory structure
        create_directory_structure(temp_dir)
        
        # Copy source files
        copy_source_files(script_dir, temp_dir)
        
        # Create ZIP archive
        output_file = os.path.abspath(args.output)
        create_zip_archive(temp_dir, output_file)
        
        print(f"Project successfully packaged to {output_file}")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main()
