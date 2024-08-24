# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# import requests
# import io
#
# app = Flask(__name__)
#
# # Constants
# ESP32_URL = "http://192.168.18.49"  # URL of your ESP32 camera
# FACE_API_URL = "http://localhost:5000/predict"  # URL of your face recognition API
#
# # Load the face detection cascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def send_image_to_api(image):
#     # Encode image as JPEG
#     _, img_encoded = cv2.imencode('.jpg', image)
#     img_bytes = img_encoded.tobytes()
#
#     # Send the image to face recognition API
#     try:
#         response = requests.post(FACE_API_URL, files={'image': ('image.jpg', img_bytes, 'image/jpeg')})
#         if response.status_code == 200:
#             return response.json()
#         else:
#             return {'name': 'Unknown'}
#     except Exception as e:
#         print(f"Error sending image to API: {e}")
#         return {'name': 'Unknown'}
#
# def generate_frames():
#     while True:
#         try:
#             response = requests.get(ESP32_URL, stream=True, timeout=5)
#             if response.status_code == 200:
#                 bytes_data = bytes()
#                 for chunk in response.iter_content(chunk_size=1024):
#                     bytes_data += chunk
#                     a = bytes_data.find(b'\xff\xd8')
#                     b = bytes_data.find(b'\xff\xd9')
#                     if a != -1 and b != -1:
#                         jpg = bytes_data[a:b + 2]
#                         bytes_data = bytes_data[b + 2:]
#                         frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#
#                         # Perform face detection
#                         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                         faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#
#                         # For simplicity, assume we only process the first detected face
#                         if len(faces) > 0:
#                             (x, y, w, h) = faces[0]
#                             face = frame[y:y + h, x:x + w]
#
#                             # Send the face image to face recognition API
#                             result = send_image_to_api(face)
#                             name = result.get('name', 'Unknown')
#
#                             # Draw rectangle and label around the face
#                             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                             cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#
#                         ret, buffer = cv2.imencode('.jpg', frame)
#                         frame_bytes = buffer.tobytes()
#                         yield (b'--frame\r\n'
#                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
