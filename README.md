# Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey) ![YOLO](https://img.shields.io/badge/YOLO-ultralytics-red) ![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)

This project is a real-time face mask detection application built with YOLOv11 for object detection and Flask for web deployment.


## Features
- Real-time face mask detection using YOLOv11
- Three-class classification: ✅ Mask, ❌ No Mask, ⚠️ Improper Mask
- Web interface with live video feed
- Confidence score display

## Installation
```bash
git clone https://github.com/sirish02/facemask-detection.git
cd facemask-detection
```

## Installing requirements
```bash
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```

## Project Structure
```bash .
├── app.py                 # Flask application
├── facemaskdetection/
│   └── yolomodel/
│       ├── yolo.py        # Detection logic
│       └── best.pt        # Pretrained weights
├── templates/
│   └── index.html         # Web interface
└── requirements.txt
```