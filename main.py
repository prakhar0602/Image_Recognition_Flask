from flask import Flask, request, jsonify
from flask_cors import CORS  
import face_recognition
import numpy as np
import os
import cv2
import pickle
import base64

app = Flask(__name__)
CORS(app)

# Define paths
DATA_DIR = 'face_data'
os.makedirs(DATA_DIR, exist_ok=True)
ENCODINGS_FILE = os.path.join(DATA_DIR, 'encodings.pkl')

def decode_image(image_data):
    image_data = image_data.split(",")[1]  # Remove the header of the base64 string
    image_data = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def save_face_encoding(face_id, encoding):
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            known_encodings = pickle.load(f)
    else:
        known_encodings = []

    known_encodings.append((face_id, encoding))

    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(known_encodings, f)

@app.route('/register', methods=['POST'])
def register_face():
    data = request.get_json()
    face_id = data.get('id')
    image_data = data.get('image')

    if not face_id or not image_data:
        return jsonify({"error": "ID and image are required"}), 400

    frame = decode_image(image_data)
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        face_encoding = face_encodings[0]
        save_face_encoding(face_id, face_encoding)
        return jsonify({"message": "Face registered successfully"}), 200
    else:
        return jsonify({"error": "No face detected"}), 400

@app.route('/authenticate', methods=['POST'])
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
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        face_encoding = face_encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            face_id = known_face_ids[best_match_index]
            return jsonify({"id": face_id}), 200
        else:
            return jsonify({"error": "Authentication failed"}), 401
    else:
        return jsonify({"error": "No face detected"}), 400


if __name__ == '__main__':
    app.run(debug=True)
