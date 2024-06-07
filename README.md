# Object Detection Project Documentation

## 1. Introduction

Welcome to the Object Detection Project documentation! This project is designed to facilitate real-time object detection and face recognition using state-of-the-art deep learning techniques. By leveraging the YOLOv5 model and pre-trained face recognition models, this project provides a robust solution for various applications, including security surveillance, image analysis, and human-computer interaction.

## 2. Features

- Real-time object detection with YOLOv5
- Face recognition using pre-trained models
- Text-to-speech feedback for detected objects
- User-friendly web-based interface via Flask application

## 3. Installation

To get started with the Object Detection Project, follow these installation instructions:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/mustafaansarii/FaceDetection-main
```

2. Navigate to the project directory:

```bash
cd object-detection
```

3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

Ensure that you have Python 3.x installed on your system.

## 4. Usage

### 4.1 Running the Flask Application

To launch the Flask application and access the object detection interface, execute the following command:

```bash
python app.py
```

This will start the Flask server, and you can access the application in your web browser at `http://localhost:5000`.

### 4.2 Running the Main Script

To perform object detection directly from the main script, run:

```bash
python main.py
```

This will execute object detection without involving the Flask framework, providing a command-line interface for interaction.

## 5. Folder Structure

The project directory follows the structure outlined below:

```
object-detection/
│
├── static/                 # Static files (e.g., images, CSS, JavaScript)
├── templates/              # HTML templates for Flask
├── app.py                  # Main Flask application
├── main.py                 # Main script for object detection
├── requirements.txt        # List of Python dependencies
└── models/                 # Directory for pre-trained models
    ├── face_recognition_model.h5
    └── yolov5m.pt
```
## Screenshots

![Screenshot 1](/screenshots/Screenshot%20from%202024-06-07%2014-29-37.png)

![Screenshot 2](/screenshots/Screenshot%20from%202024-06-07%2017-28-55.png)

## 6. Dependencies

The Object Detection Project relies on the following Python dependencies:

- torch
- tensorflow
- numpy
- opencv-python
- pyttsx3

These dependencies are listed in the `requirements.txt` file for easy installation.

## 7. Contributing

Contributions to the Object Detection Project are welcome! If you wish to contribute, please refer to the guidelines outlined in CONTRIBUTING.md.

## 8. License

This project is licensed under the MIT License. For details, see the LICENSE file.

## 9. Support

For any questions, issues, or feedback, please contact [project maintainer's email]. We appreciate your support and contribution to the project!


