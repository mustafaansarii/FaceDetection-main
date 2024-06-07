from flask import Flask, render_template, Response, jsonify
import torch
import tensorflow as tf
import numpy as np
import cv2
import pyttsx3

app = Flask(__name__)

# Load the object detection model (YOLOv5)
object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True, trust_repo=True)
object_detection_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the face recognition model
face_recognition_model = tf.keras.models.load_model('face_recognition_model.h5')
face_recognition_class_names = ['Person1', 'Person2', 'Person3', 'Person4', 'Person5', 'Person6', 'Person7', 'Person8', 'Person9', 'Person10']

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess a single image for face recognition
def preprocess_image(image):
    img = cv2.resize(image, (100, 100))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to draw bounding boxes for object detection
def draw_boxes(img, results):
    detected_objects = []
    for result in results.xyxy[0]:
        xmin, ymin, xmax, ymax = map(int, result[:4])
        label = object_detection_model.names[int(result[5])]
        confidence = result[4]
        detected_objects.append(label)
        color = (0, 255, 0)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return detected_objects

# Function to perform object detection and text-to-speech
def object_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1080, 720))
        results = object_detection_model(frame)
        detected_objects = draw_boxes(frame, results)

        if detected_objects:
            objects_str = ', '.join(detected_objects)
            text = f"I can see {objects_str}."
            engine.say(text)
            engine.runAndWait()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to perform face recognition
def face_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1080, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            preprocessed_face = preprocess_image(face_roi)
            predictions = face_recognition_model.predict(preprocessed_face)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_person = face_recognition_class_names[predicted_class[0]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, predicted_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_flask_app')
def start_flask_app():
    return jsonify(success=True)

@app.route('/object_detection_feed')
def object_detection_feed():
    return Response(object_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_recognition_feed')
def face_recognition_feed():
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
