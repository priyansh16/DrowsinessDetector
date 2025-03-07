# Drowsiness Detection System

## Overview
The **Drowsiness Detection System** is a real-time application designed to detect drowsiness in individuals based on eye state recognition. Using a deep learning model trained on eye images, the system can identify closed eyes and trigger an alarm to alert the individual. This project is implemented using **TensorFlow, OpenCV, and Pygame**.

## Features
- Real-time eye detection using OpenCV.
- Convolutional Neural Network (CNN) model trained to classify eye states (open/closed).
- Alarm system that triggers when drowsiness is detected.
- Adjustable threshold for drowsiness detection duration.

## Technologies Used
- **Python** (Primary language)
- **TensorFlow/Keras** (For CNN model training)
- **OpenCV** (For face and eye detection)
- **Pygame** (For playing alert sounds)
- **Matplotlib** (For visualizing training history)

## Project Structure
```bash
Drowsiness-Detection-System/
│-- src/
│   ├── DrowsinessDetector.py  # Drowsiness detection logic
│   ├── train_eye_cnn.py       # Trains the cnn model for detection
│-- models/
│   ├── best_model.keras       # Trained CNN model classifying the eyes
│-- dataset/                   # Dataset for training the model
│-- main.py                     # Parse user input and detect drowsiness      
│-- README.md                   
│-- requirements.txt             # Required dependencies to run project.
```

## CNN Model Design
The model follows a **Convolutional Neural Network (CNN) architecture** with the following layers:
```text
1. Conv2D Layer (16 filters, 3x3 kernel) + ReLU activation + L2 Regularization
2. MaxPooling2D Layer (2x2 pool size)
3. BatchNormalization + Dropout (0.2)
4. Conv2D Layer (32 filters, 3x3 kernel) + ReLU activation + L2 Regularization
5. MaxPooling2D Layer (2x2 pool size)
6. BatchNormalization + Dropout (0.2)
7. Flatten Layer
8. Dense Layer (32 neurons, ReLU activation, L2 Regularization)
9. BatchNormalization + Dropout (0.3)
10. Output Dense Layer (1 neuron, Sigmoid activation for binary classification)
```

## Installation
### Prerequisites
Ensure you have Python **3.8+** installed. Install required dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Drowsiness Detection System
Execute the following command to run the real-time detection system with default parameters:
```bash
python main.py 
```

Execute the following commad with user input to run the real-time detection with user input:
```bash
python main.py --model models/best_model.keras --alarm Alert.wav --drowsy_time 2.0
```

### Training the CNN Model
To train the model on a dataset of eye images:
```bash
python train_eye_cnn.py
```
This will save the trained model as `models/best_model.keras` and generate training history plots.

### Stopping Detection
Press **'q'** to exit the detection process.

## Applications
- **Driver Monitoring**: Helps prevent accidents due to driver drowsiness.
- **Workplace Safety**: Ensures alertness in critical job roles.
- **Smart Surveillance**: Monitors drowsiness in security personnel.

## Acknowledgements
This project leverages open-source libraries and **Haar cascade classifiers** for face and eye detection provided by OpenCV.

## License
This project is licensed under the **MIT License**.

