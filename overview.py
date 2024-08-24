from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import requests
import threading
import time

app = Flask(__name__)

# ESP32 stream URL
ESP32_URL = "http://192.168.18.49"  # Replace with your ESP32's IP and port

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="best_siamese_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Similarity threshold
SIMILARITY_THRESHOLD = 0.7  # Adjust this value as needed

# Global variables for thread communication
frame = None
frame_ready = False
current_name = "Unknown"
current_score = None
face_location = None

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))  # Adjust size as per your model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    image = np.repeat(image, 32, axis=0)
    return image

# Function to get embedding
def get_embedding(image):
    preprocessed = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], preprocessed)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# Function to compare embeddings
def compare_embeddings(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

# Load known face embeddings
known_face = cv2.imread("73460658_198415224497784_7124123628957859840_n.jpg")
if known_face is None:
    raise FileNotFoundError("Known face image not found")
known_embeddings = {
    "bishwash": get_embedding(known_face)
}

# Function to recognize face
def recognize_face(face_image):
    embedding = get_embedding(face_image)
    min_distance = float('inf')
    recognized_name = "Unknown"
    for name, known_embedding in known_embeddings.items():
        distance = compare_embeddings(embedding, known_embedding)
        if distance < min_distance:
            min_distance = distance
            recognized_name = name

    similarity_score = 1 - min_distance  # Convert distance to similarity
    if similarity_score >= SIMILARITY_THRESHOLD:
        return recognized_name, similarity_score
    else:
        return "Unknown", similarity_score

# Thread function to capture frames
def capture_frames():
    global frame, frame_ready
    while True:
        response = requests.get(ESP32_URL, stream=True)
        if response.status_code == 200:
            bytes_data = bytes()
            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b + 2]
                    bytes_data = bytes_data[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    frame_ready = True

# Thread function to process frames
def process_frames():
    global frame, frame_ready, current_name, current_score, face_location
    frame_count = 0
    while True:
        if frame_ready:
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Process only the first detected face
                face_location = (x, y, w, h)
                if frame_count % 30 == 0:  # Process every 30th frame
                    face_image = frame[y:y+h, x:x+w]
                    current_name, current_score = recognize_face(face_image)
            else:
                face_location = None
                current_name = "Unknown"
                current_score = None
            frame_ready = False

# Start threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
capture_thread.start()
process_thread.start()

def generate_frames():
    global frame, current_name, current_score, face_location
    while True:
        if frame is not None:
            display_frame = frame.copy()
            if face_location is not None:
                (x, y, w, h) = face_location
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if current_name != "Unknown" and current_score is not None:
                    label = f"{current_name} (Similarity: {current_score:.2f})"
                elif current_score is not None:
                    label = f"Unknown (Similarity: {current_score:.2f})"
                else:
                    label = "Detecting..."
                cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)