# Lunar Image Enhancement Project

This project provides tools for enhancing dark regions in lunar images using both lightweight and deep learning approaches.

## Features

- **Lightweight Version**: Uses classical image processing techniques for quick and efficient enhancement
- **Web Interface**: Streamlit-based web application for easy image upload and enhancement
- **Demo Mode**: Includes sample lunar images to demonstrate the enhancement capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lunar-image-enhancement.git
cd lunar-image-enhancement
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
cd frontend
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Upload a lunar image or use the demo image to see the enhancement in action

## Lightweight Model Details

The lightweight version uses a combination of image processing techniques:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral filtering
- Unsharp masking
- Gamma correction

## Project Structure

```
lunar-image-enhancement/
├── frontend/
│   └── app.py              # Streamlit web interface
├── src/
│   ├── lightweight_model.py # Lightweight enhancement implementation
│   └── destripenet.py      # Full DestripeNet implementation
├── results/                # Output directory for enhanced images
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 