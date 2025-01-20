# PCB Fault Detection 

This project is designed to detect and describe defects in printed circuit boards (PCBs) using a YOLOv5 deep learning model. It identifies common defects like missing holes, open circuits, and short circuits, and provides bounding boxes with confidence levels for each detection.

## Features
- **Defect Detection**: Identifies common PCB defects such as missing holes, open circuits, and short circuits.
- **YOLOv5 Integration**: Utilizes the YOLOv5 model for real-time object detection.
- **Confidence Levels**: Provides confidence scores for detected defects.
- **Annotated Output**: Generates an image with bounding boxes and labels for detected defects.
- **Web Interface**: A local web interface for uploading PCB images and viewing annotated results.

## Requirements
- Python 3.12.8
- CUDA-compatible GPU (e.g., NVIDIA GeForce RTX 4090)
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd pcb_detection
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python3 -m venv yolov5-env
   source yolov5-env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv5 Model**:
   Ensure the YOLOv5 model files are present in the `model` directory.

## Usage

1. **Run the Application**:
   ```bash
   python3 appVision.py
   ```

2. **Access the Web Interface**:
   - Open your browser and go to the local URL displayed in the terminal (e.g., `http://127.0.0.1:7860`).

3. **Upload PCB Images**:
   - Upload an image of the PCB to analyze.
   - View the annotated image with defect descriptions and confidence levels.

## Directory Structure
```
.
├── appVision.py          # Main application file
├── model                 # Directory containing YOLOv5 model files
├── requirements.txt      # Python dependencies
├── temp_annotated.jpg    # Temporary annotated image file
├── yolov5-env            # Virtual environment directory
```

## Output
- **Annotated Image**: The annotated image with detected defects and bounding boxes is saved as `temp_annotated.jpg`.
- **Terminal Log**: Confidence levels and defect descriptions are logged in the terminal.

## Troubleshooting
- **CUDA Errors**: Ensure that your GPU supports CUDA and the correct drivers are installed.
- **Model Not Found**: Verify that the YOLOv5 model files are in the `model` directory.
- **Dependencies**: Reinstall dependencies using `pip install -r requirements.txt` if issues arise.

## Future Improvements
- Add support for public links to access the web interface remotely.
- Increase the range of detectable defects.
- Optimize the model for faster inference.

## Acknowledgments
This project uses the YOLOv5 object detection model, available at [Ultralytics](https://github.com/ultralytics/yolov5).

## License
This project is licensed under the MIT License.
