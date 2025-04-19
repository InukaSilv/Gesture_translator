# ASL Gesture Translator

An advanced real-time American Sign Language (ASL) recognition system that combines deep learning with computer vision techniques to interpret hand gestures. Features MediaPipe-powered finger tracking and a robust CNN model for accurate classification.

## Key Features

- ✋ **Real-time hand tracking** using MediaPipe
- 🧠 **Deep learning model** for 26 ASL letters (A-Z)
- 📊 **Data augmentation** for improved generalization
- ⚡ **Optimized performance** with prediction smoothing
- 📱 **Webcam compatibility** for easy deployment

## Requirements
Python 3.8+  
Webcam  
NVIDIA GPU (recommended)

## Installation

1. Clone the repository:  
`git clone https://github.com/yourusername/asl-translator.git`  
`cd asl-translator`

2. Install dependencies:  
`pip install -r requirements.txt`

3. Run the ASL translator:  
`python predict.py`  
Show ASL letters to your webcam  
Press Q to quit

**Future Improvements**  
Add sentence-level interpretation  
Support for numbers (0-9)  
3D gesture recognition  
Mobile app deployment  

This project is licensed under the MIT License - see the LICENSE file for details.
