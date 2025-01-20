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
        name = request.form['name']

        # If an image is uploaded
        if 'uploaded_image' in request.files and request.files['uploaded_image'].filename != '':
            uploaded_file = request.files['uploaded_image']
            if uploaded_file:
                image_path = os.path.join(FACES_DIR, f"{name}.jpg")
                encoding_path = os.path.join(FACES_DIR, f"{name}_encoding.npy")
                uploaded_file.save(image_path)

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



import time  # For the timer functionality

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        # Reload encodings dynamically
        global known_encodings, known_names
        known_encodings, known_names = load_encodings(FACES_DIR)

        # If a video is uploaded
        if 'uploaded_video' in request.files and request.files['uploaded_video'].filename != '':
            uploaded_file = request.files['uploaded_video']
            if uploaded_file:
                video_path = os.path.join(FACES_DIR, 'uploaded_video.mp4')
                uploaded_file.save(video_path)

                cap = cv2.VideoCapture(video_path)
                attendance_data = process_video_or_camera_feed(cap, known_encodings, known_names)
                cap.release()

                # Save attendance to file
                if attendance_data:
                    pd.DataFrame(attendance_data).to_csv(ATTENDANCE_FILE, mode='a', index=False, header=False)
                # Redirect to view attendance
                return redirect(url_for('view_attendance'))
        # If capturing live from camera
        elif 'capture' in request.form:
            print("Attempting to open camera...")
            cap = cv2.VideoCapture(0)  # Default camera index
            if not cap.isOpened():
                print("Failed to open camera.")
                return render_template('attendance.html', error="Camera could not be opened.")
            print("Camera opened successfully.")

            attendance_data = process_video_or_camera_feed(cap, known_encodings, known_names)
            cap.release()

            # Save attendance to file
            if attendance_data:
                pd.DataFrame(attendance_data).to_csv(ATTENDANCE_FILE, mode='a', index=False, header=False)
            return render_template('attendance.html', success="Live attendance recorded successfully!")

    return render_template('attendance.html')

# Route: View Attendance
@app.route('/view-attendance')
def view_attendance():
    try:
        # Load attendance data from CSV
        if os.path.exists(ATTENDANCE_FILE):
            # Ensure proper headers while reading
            attendance_data = pd.read_csv(ATTENDANCE_FILE, header=None, names=["Timestamp", "Name"])
            print('attendance_data', attendance_data)
            records = attendance_data.to_dict(orient='records')  # Convert to list of dictionaries
        else:
            records = []  # Empty if the file doesn't exist
    except Exception as e:
        print(f"Error loading attendance: {e}")
        records = []

    return render_template('view_attendance.html', records=records)


import hashlib

ATTENDANCE_FILE = 'attendance.csv'
FACES_DIR = 'faces'
target_width, target_height = 960, 540  # Desired window size

def process_video_or_camera_feed(cap, known_encodings, known_names):
    """
    Process the given video capture (live camera or uploaded video) for attendance.
    Saves attendance only when match percentage is above 55%.
    """
    attendance_data = []
    last_attendance_time = {}
    start_time = time.time()  # Record the start time
    save_interval = 300  # 5 minutes in seconds

    def scale_and_pad(frame, target_width, target_height):
        """Resize and pad the frame while maintaining aspect ratio."""
        original_height, original_width = frame.shape[:2]
        scale = min(target_width / original_width, target_height / original_height)
        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)

        resized_frame = cv2.resize(frame, (resized_width, resized_height))
        padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - resized_width) // 2
        y_offset = (target_height - resized_height) // 2
        padded_frame[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized_frame

        return padded_frame, resized_frame, scale, x_offset, y_offset

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting loop.")
            break

        padded_frame, resized_frame, scale, x_offset, y_offset = scale_and_pad(frame, target_width, target_height)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            matches = distances <= 0.6  # Threshold for recognizing faces
            name = "Unknown"
            match_percentage = 0

            if any(matches):
                # Get the best match index
                best_match_index = np.argmin(distances)
                match_percentage = (1 - distances[best_match_index]) * 100  # Convert distance to percentage
                if match_percentage > 55:  # Only consider matches above 55%
                    name = known_names[best_match_index]

                    # Handle known face attendance
                    current_time = datetime.now()
                    current_hour = current_time.replace(minute=0, second=0, microsecond=0)
                    if name not in last_attendance_time or last_attendance_time[name] < current_hour:
                        last_attendance_time[name] = current_hour
                        attendance_data.append({'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'), 'Name': name})
                        print(f"Attendance recorded for {name} ({match_percentage:.2f}%) at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Draw bounding box and name with match percentage on the resized frame
            (top, right, bottom, left) = face_location
            top = int(top * scale) + y_offset
            right = int(right * scale) + x_offset
            bottom = int(bottom * scale) + y_offset
            left = int(left * scale) + x_offset
            color = (0, 255, 0) if match_percentage > 55 else (0, 0, 255)  # Green for valid, Red otherwise
            cv2.rectangle(padded_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                padded_frame,
                f"{name} ({match_percentage:.2f}%)",
                (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        # Save attendance every 5 minutes
        if time.time() - start_time >= save_interval:
            print("Saving attendance data...")
            pd.DataFrame(attendance_data).to_csv(ATTENDANCE_FILE, mode='a', index=False, header=False)
            attendance_data.clear()  # Clear data after saving
            start_time = time.time()  # Reset the timer

        # Display the padded frame
        cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)  # Make the window resizable
        cv2.resizeWindow('Attendance', target_width, target_height)
        cv2.imshow('Attendance', padded_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
            break

    cv2.destroyAllWindows()
    print("Processing complete. Closing video feed.")
    return attendance_data



# def restart_flask_server():
#     time.sleep(1)  # Allow the current request to finish
#     print("Restarting Flask server...")
#     os.execv(sys.executable, ['python'] + sys.argv)

# @app.after_request
# def reload_app(response):
#     if request.method == 'POST' and request.endpoint in ['attendance', 'register']:
#         print("Scheduling Flask server restart...")
#         threading.Thread(target=restart_flask_server).start()
#     return response





if __name__ == '__main__':
    app.run(debug=True)


