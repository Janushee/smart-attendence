from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import os
import sys
import threading
import time

app = Flask(__name__)

# Directories for storing data
FACES_DIR = 'faces'
ATTENDANCE_FILE = 'attendance.csv'
os.makedirs(FACES_DIR, exist_ok=True)

# Load existing encodings
def load_encodings(encodings_path):
    encodings, names = [], []
    for file in os.listdir(encodings_path):
        if file.endswith("_encoding.npy"):
            name = file.split('_')[0]
            encoding = np.load(os.path.join(encodings_path, file))
            encodings.append(encoding)
            names.append(name)
    return encodings, names

# Global variables for known encodings and names
known_encodings, known_names = load_encodings(FACES_DIR)

# Route: Home
@app.route('/')
def index():
    return render_template('index.html')

# Route: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # If the user chooses to upload an image
        if 'uploaded_image' in request.files and request.files['uploaded_image'].filename != '':
            uploaded_file = request.files['uploaded_image']
            name = request.form['name']
            if uploaded_file:
                image_path = os.path.join(FACES_DIR, f"{name}.jpg")
                encoding_path = os.path.join(FACES_DIR, f"{name}_encoding.npy")
                
                # Save uploaded image
                uploaded_file.save(image_path)

                # Process the uploaded image
                frame = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(frame)

                if encodings:
                    encoding = encodings[0]
                    np.save(encoding_path, encoding)
                    return render_template('index.html', success=f"Face registered for {name}!")
                else:
                    return render_template('register.html', error="No face detected in the uploaded image.")

        # If the user chooses to capture an image from the camera
        elif 'capture' in request.form:
            name = request.form['name']
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                return render_template('register.html', error="Camera could not be opened.")

            saved = False
            while not saved:
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_frame)

                if encodings:
                    encoding = encodings[0]
                    face_image_path = os.path.join(FACES_DIR, f"{name}.jpg")
                    encoding_path = os.path.join(FACES_DIR, f"{name}_encoding.npy")
                    cv2.imwrite(face_image_path, frame)
                    np.save(encoding_path, encoding)
                    saved = True

            cap.release()
            cv2.destroyAllWindows()

            return render_template('index.html', success=f"Face registered for {name}!")

    return render_template('register.html')


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        # Reload encodings dynamically
        global known_encodings, known_names
        known_encodings, known_names = load_encodings(FACES_DIR)

        print("Attempting to open camera...")
        cap = cv2.VideoCapture(0)  # Default camera index
        if not cap.isOpened():
            print("Failed to open camera.")
            return render_template('attendance.html', error="Camera could not be opened.")
        print("Camera opened successfully.")

        attendance_data = []
        last_attendance_time = {}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_names[first_match_index]

                    if name != "Unknown":
                        current_time = datetime.now()
                        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
                        if name not in last_attendance_time or last_attendance_time[name] < current_hour:
                            last_attendance_time[name] = current_hour
                            attendance_data.append({'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'), 'Name': name})

                    (top, right, bottom, left) = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow('Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            print("Releasing camera...")
            cap.release()
            del cap
            cv2.destroyAllWindows()
            print("Camera released.")

        # Save to attendance file
        pd.DataFrame(attendance_data).to_csv(ATTENDANCE_FILE, mode='a', index=False, header=False)

        # Redirect to homepage
        return redirect(url_for('index'))

    return render_template('attendance.html')


def restart_flask_server():
    time.sleep(1)  # Allow the current request to finish
    print("Restarting Flask server...")
    os.execv(sys.executable, ['python'] + sys.argv)

@app.after_request
def reload_app(response):
    if request.method == 'POST' and request.endpoint in ['attendance', 'register']:
        print("Scheduling Flask server restart...")
        threading.Thread(target=restart_flask_server).start()
    return response





if __name__ == '__main__':
    app.run(debug=True)
