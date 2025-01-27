from flask import Flask, Response, render_template, request, redirect, url_for, flash
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
from flask_socketio import SocketIO, emit
from collections import defaultdict
import base64
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=1e8, ping_timeout=120, ping_interval=25)

# Directories for storing data
FACES_DIR = 'faces'
ATTENDANCE_FILE = 'attendance.csv'
os.makedirs(FACES_DIR, exist_ok=True)


app = Flask(__name__)

# MJPEG URL (for camera feed)
MJPEG_URL = 'http://192.168.11.130:8080/video'

# Video Capture
capture = cv2.VideoCapture(MJPEG_URL)

def load_encodings(encodings_path):
    encodings, names = [], []
    for file in os.listdir(encodings_path):
        if file.endswith("_encoding.npy"):
            name = file.split('_')[0]
            encoding = np.load(os.path.join(encodings_path, file))
            encodings.append(encoding)
            names.append(name)
    return encodings, names

known_encodings, known_names = load_encodings(FACES_DIR)

# Track the last time a person was recorded
last_recorded_time = defaultdict(lambda: datetime.min)

# Frame rate for face recognition (process every nth frame)
FACE_RECOGNITION_INTERVAL = 5  # Process every 5th frame

# Reduce video resolution for faster processing
frame_width = 640  # Adjust the resolution for better performance
frame_height = 480

def load_encodings(encodings_path):
    encodings, names = [], []
    for file in os.listdir(encodings_path):
        if file.endswith("_encoding.npy"):
            name = file.split('_')[0]
            encoding = np.load(os.path.join(encodings_path, file))
            encodings.append(encoding)
            names.append(name)
    return encodings, names

known_encodings, known_names = load_encodings(FACES_DIR)

# Track the last time a person was recorded
last_recorded_time = defaultdict(lambda: datetime.min)

# Frame rate for face recognition (process every nth frame)
FACE_RECOGNITION_INTERVAL = 5  # Process every 5th frame

# Reduce video resolution for faster processing
frame_width = 640  # Adjust the resolution for better performance
frame_height = 480

# Stores recognized faces and their locations for continuity
recognized_faces_data = {}


# Generate video frames with face recognition
def generate_frames():
    capture = cv2.VideoCapture(MJPEG_URL)

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"
            match_percentage = 0

            if any(distances <= 0.5):  # Recognition threshold
                best_match_index = np.argmin(distances)
                name = known_names[best_match_index]
                match_percentage = (1 - distances[best_match_index]) * 100
                recognized_faces.append({'name': name, 'match_percentage': match_percentage})

                # Log attendance if new person or the same person appeared after 30 seconds
                now = datetime.now()
                if name not in last_recorded_time or (now - last_recorded_time[name]).total_seconds() > 30:
                    last_recorded_time[name] = now
                    attendance_data = {
                        "Timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
                        "Name": name,
                        "Method": "Live-stream",  # Customize to the camera name
                    }

                    # Append to CSV
                    df = pd.DataFrame([attendance_data])
                    df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE))

                    # Log to terminal
                    print(f"Recorded Attendance for {name} at {attendance_data['Timestamp']}")

                    # Emit alert to frontend
                    socketio.emit('attendance_captured_live', {
                        'name': name,
                        'timestamp': attendance_data["Timestamp"]
                    })

            # Draw the bounding box and name with match percentage
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({match_percentage:.2f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode the frame and send to client
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    capture.release()

@app.route('/video-feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Ensure `data` contains both frame and camera_name
        if not isinstance(data, dict) or 'frame' not in data or 'camera_name' not in data:
            emit('response_frame', {'error': 'Invalid data received. Expected frame and camera_name.'})
            return

        frame_data = data['frame']
        camera_name = data['camera_name']

        try:
            image_data = base64.b64decode(frame_data.split(',')[1])
        except Exception as e:
            emit('response_frame', {'error': f'Base64 decoding error: {str(e)}'})
            return        
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('response_frame', {'error': 'Invalid frame received'})
            return

        # Downscale the frame to reduce processing load but maintain aspect ratio
        original_height, original_width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (320, int(320 * original_height / original_width)))

        # Process the resized frame for face recognition
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"
            match_percentage = 0

            if any(distances <= 0.6):  # Recognition threshold
                best_match_index = np.argmin(distances)
                name = known_names[best_match_index]
                match_percentage = (1 - distances[best_match_index]) * 100
                recognized_faces.append({'name': name, 'match_percentage': match_percentage})

                if match_percentage > 50:
                    now = datetime.now()
                    if name not in last_recorded_time or (now - last_recorded_time[name]).total_seconds() > 30:
                        last_recorded_time[name] = now

                        # Write to CSV
                        attendance_data = {
                            "Timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
                            "Name": name,
                            "Method": camera_name,  # Use the camera name here
                        }
                        pd.DataFrame([attendance_data]).to_csv(
                            ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE)
                        )

                        # Send alert to the client
                        emit('attendance_captured', {
                            'name': name,
                            'timestamp': attendance_data["Timestamp"],
                            'camera_name': camera_name
                        })

            # Draw bounding box and name (convert back to the original frame's scale)
            top, right, bottom, left = face_location
            scale_x = original_width / 320
            scale_y = original_height / int(320 * original_height / original_width)
            top, right, bottom, left = [
                int(coord * scale_y if i % 2 == 0 else coord * scale_x)
                for i, coord in enumerate([top, right, bottom, left])
            ]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({match_percentage:.2f}%)",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Encode the processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_frame = base64.b64encode(buffer).decode('utf-8')

        # Send the processed frame and recognition data back to the client
        emit('response_frame', {
            'frame': f"data:image/jpeg;base64,{processed_frame}",
            'recognized_faces': recognized_faces
        })

    except Exception as e:
        emit('response_frame', {'error': str(e)})



# Route: Home
@app.route('/')
def index():
    return render_template('index.html')

# Route: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

# Route: Attendance
@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

# Route: View Attendance
@app.route('/view-attendance')
def view_attendance():
    try:
        # Load attendance data from CSV
        if os.path.exists(ATTENDANCE_FILE):
            attendance_data = pd.read_csv(ATTENDANCE_FILE, header=None, names=["Timestamp", "Name", "Method"])
            records = attendance_data.to_dict(orient='records')  # Convert to list of dictionaries
        else:
            records = []  # Empty if the file doesn't exist
    except Exception as e:
        print(f"Error loading attendance: {e}")
        records = []

    return render_template('view_attendance.html', records=records)

if __name__ == '__main__':
    socketio.run(app, debug=True,allow_unsafe_werkzeug=True)
