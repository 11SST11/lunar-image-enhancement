#!/usr/bin/env python3
"""
Lunar Image Enhancement Project
VSCode Setup Guide

This file provides instructions for setting up and running the project in VSCode.
"""

# VSCode Setup Guide for Lunar Image Enhancement Project

## Prerequisites

1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install [Python](https://www.python.org/downloads/) (version 3.8 or higher)
3. Install [Git](https://git-scm.com/downloads) (optional, for version control)

## Setup Steps

### 1. Clone or Download the Repository

**Option 1: Using Git**
```bash
git clone https://github.com/yourusername/lunar_image_enhancement.git
cd lunar_image_enhancement
```

**Option 2: Download ZIP**
- Download the ZIP file from the repository
- Extract it to your preferred location

### 2. Open the Project in VSCode

```bash
code /path/to/lunar_image_enhancement
```

Or open VSCode, then:
- File > Open Folder
- Navigate to the lunar_image_enhancement directory
- Click "Open"

### 3. Set Up a Virtual Environment

Open a terminal in VSCode (Terminal > New Terminal) and run:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Python Interpreter in VSCode

- Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
- Type "Python: Select Interpreter"
- Select the interpreter from your virtual environment (usually has "venv" in the path)

### 5. Run the Application

#### Option 1: Run the Streamlit App

In the VSCode terminal:
```bash
streamlit run frontend/app.py
```

#### Option 2: Run the Lightweight Model Demo

In the VSCode terminal:
```bash
python src/lightweight_model.py
```

### 6. Debugging in VSCode

To debug the application:

1. Set breakpoints by clicking in the gutter (left margin) of the code editor
2. Press F5 or click the Run > Start Debugging menu
3. Select "Python File" as the debug configuration
4. The debugger will stop at your breakpoints

### 7. Recommended VSCode Extensions

Install these extensions for a better development experience:

- Python (Microsoft)
- Pylance
- Python Indent
- autoDocstring
- Jupyter
- GitLens (if using Git)

### 8. Running Tests

To run tests in VSCode:

```bash
# Run a specific test file
python -m unittest src/test.py

# Run all tests
python -m unittest discover
```

### 9. Troubleshooting

If you encounter issues:

1. **ImportError**: Make sure your virtual environment is activated and all dependencies are installed
2. **ModuleNotFoundError**: Check that you're running the commands from the project root directory
3. **Permission issues**: Make sure scripts are executable (`chmod +x script.py` on Linux/macOS)

For more help, refer to the main README.md file or open an issue on the repository.
