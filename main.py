import tensorflow as tf
import numpy as np
import cv2

face_recognition_model = tf.keras.models.load_model('face_recognition_model.h5')
face_recognition_class_names = ['Person1', 'Person2', 'Person3', 'Person4', 'Person5', 'Person6', 'Person7', 'Person8', 'Person9', 'Person10']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess a single image for face recognition
def preprocess_image(image):
    img = cv2.resize(image, (100, 100))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

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

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

face_recognition()
