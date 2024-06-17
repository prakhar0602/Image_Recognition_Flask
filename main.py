from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import os
import pickle
import base64

app = Flask(__name__)
CORS(app)

# Define paths
DATA_DIR = 'face_data'
os.makedirs(DATA_DIR, exist_ok=True)
ENCODINGS_FILE = os.path.join(DATA_DIR, 'encodings.pkl')
TRAIN_DATA_DIR = 'lfw-deepfunneled'  # Folder containing training images

# Load pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def decode_image(image_data):
    image_data = image_data.split(",")[1]  # Remove the header of the base64 string
    image_data = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def get_face_encodings(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return None, None

    face_encodings = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))  # Resize face to consistent size for encoding
        encoding = np.mean(face, axis=(0, 1))
        face_encodings.append(encoding)

    return faces, face_encodings

def save_face_encoding(face_id, encoding):
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            known_encodings = pickle.load(f)
    else:
        known_encodings = []

    known_encodings.append((face_id, encoding))

    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(known_encodings, f)

def read_images_from_folder(folder_path):
    face_encodings = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(root, filename)
                frame = cv2.imread(image_path)

                # Get face encodings
                _, encodings = get_face_encodings(frame)

                if encodings:
                    for i, encoding in enumerate(encodings):
                        # Use relative path from the root folder as ID
                        relative_path = os.path.relpath(image_path, folder_path)
                        face_encodings.append((relative_path, encoding))

    return face_encodings

@app.route('/train', methods=['POST'])
@cross_origin(origins='http://localhost:5173')
def train_faces():
    print('start')
    face_encodings = read_images_from_folder(TRAIN_DATA_DIR)
    print('done')
    if face_encodings:
        print('start2')
        for filename, encoding in face_encodings:
            print(filename)
            save_face_encoding(filename, encoding)
        print('end')
        
        return jsonify({"message": "Training completed successfully"}), 200
    else:
        return jsonify({"error": "No faces detected or encoding failed for training images"}), 400

@app.route('/register', methods=['POST'])
@cross_origin(origins='http://localhost:5173')
def register_face():
    data = request.get_json()
    face_id = data.get('id')
    image_data = data.get('image')

    if not face_id or not image_data:
        return jsonify({"error": "ID and image are required"}), 400

    frame = decode_image(image_data)

    # Get face encodings
    _, encodings = get_face_encodings(frame)

    if encodings:
        for i, encoding in enumerate(encodings):
            save_face_encoding(f"{face_id}_{i}", encoding)
        return jsonify({"message": "Faces registered successfully"}), 200
    else:
        return jsonify({"error": "No faces detected or encoding failed"}), 400

@app.route('/authenticate', methods=['POST'])
@cross_origin(origins='http://localhost:5173')
def authenticate_face():
    if not os.path.exists(ENCODINGS_FILE):
        return jsonify({"error": "No faces registered. Please register faces first."}), 400

    with open(ENCODINGS_FILE, 'rb') as f:
        known_encodings = pickle.load(f)

    known_face_ids = [enc[0] for enc in known_encodings]
    known_face_encodings = [enc[1] for enc in known_encodings]

    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({"error": "Image is required"}), 400

    frame = decode_image(image_data)

    # Get face encodings
    _, encodings = get_face_encodings(frame)

    if encodings:
        # Compare each detected face encoding with known encodings
        results = []
        for encoding in encodings:
            distances = np.linalg.norm(np.array(known_face_encodings) - encoding, axis=1)
            min_distance_index = np.argmin(distances)

            if distances[min_distance_index] < 100:  # Adjust threshold as needed
                face_id = known_face_ids[min_distance_index]
                return jsonify({"id": face_id}), 200
            else:
                results.append({"error": "Authentication failed"})

        # If no matching face found
        return jsonify({"error": "No matching face found"}), 401

    else:
        return jsonify({"error": "No faces detected or encoding failed"}), 400

if __name__ == '__main__':
    app.run(debug=True)
