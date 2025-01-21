from flask import Flask, render_template, request, redirect, url_for, flash
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
import base64
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")



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

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Process incoming video frames from the client.
    """
    try:
        # Decode the base64 image
        image_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('response_frame', {'error': 'Invalid frame received'})
            return

        # Process the frame for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_recognized = False
        recognized_name = None

        for face_encoding, face_location in zip(face_encodings, face_locations):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"
            match_percentage = 0

            if any(distances <= 0.6):  # Recognition threshold
                best_match_index = np.argmin(distances)
                name = known_names[best_match_index]
                match_percentage = (1 - distances[best_match_index]) * 100
                face_recognized = True
                recognized_name = name

            # Draw bounding box and name
            top, right, bottom, left = face_location
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

        # Send the processed frame and recognition status back to the client
        emit('response_frame', {
            'frame': f"data:image/jpeg;base64,{processed_frame}",
            'faceRecognized': face_recognized,
            'name': recognized_name
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
    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            flash("Please enter a name.", "danger")
            return redirect(url_for('register'))

        captured_image = request.form.get('captured_image')  # Get Base64 image from form

        if captured_image:
            try:
                # Decode the Base64 image but do not save it yet
                image_data = base64.b64decode(captured_image.split(',')[1])

                # Load the image into memory to check its validity
                nparr = np.frombuffer(image_data, np.uint8)
                temp_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if temp_image is None:  # Check if image decoding failed
                    flash("Error: Captured image is invalid or unavailable.", "danger")
                    return redirect(url_for('register'))

                # Process the valid image
                frame = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(frame)

                if encodings:
                    encoding = encodings[0]
                    image_path = os.path.join(FACES_DIR, f"{name}.jpg")
                    encoding_path = os.path.join(FACES_DIR, f"{name}_encoding.npy")

                    # Save the image and encoding only if a face is detected
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    np.save(encoding_path, encoding)

                    flash(f"Face registered for {name}!", "success")
                    return redirect(url_for('index'))
                else:
                    flash("No face detected in the captured image.", "danger")
                    return redirect(url_for('register'))
            except Exception as e:
                flash(f"Error processing the captured image: {e}", "danger")
                return redirect(url_for('register'))

        # Handle uploaded image
        # if 'uploaded_image' in request.files and request.files['uploaded_image'].filename != '':
        #     uploaded_file = request.files['uploaded_image']
        #     try:
        #         if uploaded_file:
        #             # Temporarily load the image into memory to validate it
        #             image_path = os.path.join(FACES_DIR, f"{name}.jpg")
        #             uploaded_file.save(image_path)

        #             frame = face_recognition.load_image_file(image_path)
        #             encodings = face_recognition.face_encodings(frame)

        #             if encodings:
        #                 encoding = encodings[0]
        #                 encoding_path = os.path.join(FACES_DIR, f"{name}_encoding.npy")
        #                 np.save(encoding_path, encoding)
        #                 flash(f"Face registered for {name}!", "success")
        #                 return redirect(url_for('index'))
        #             else:
        #                 os.remove(image_path)  # Remove the invalid file
        #                 flash("No face detected in the uploaded image.", "danger")
        #                 return redirect(url_for('register'))
        #     except Exception as e:
        #         flash(f"Error processing the uploaded image: {e}", "danger")
        #         return redirect(url_for('register'))

        # If no valid input is provided
        flash("Please capture an image or upload one.", "danger")
        return redirect(url_for('register'))

    return render_template('register.html')



import time  # For the timer functionality

# @app.route('/attendance', methods=['GET', 'POST'])
# def attendance():
#     if request.method == 'POST':
#         # Reload encodings dynamically
#         global known_encodings, known_names
#         known_encodings, known_names = load_encodings(FACES_DIR)

#         # Handle the captured image
#         captured_image = request.form.get('captured_image')
#         if captured_image:
#             # Decode the Base64 image and save it as a file
#             try:
#                 image_data = base64.b64decode(captured_image.split(',')[1])
#                 temp_image_path = os.path.join(FACES_DIR, "temp.jpg")
#                 with open(temp_image_path, 'wb') as f:
#                     f.write(image_data)

#                 # Process the saved image
#                 frame = face_recognition.load_image_file(temp_image_path)
#                 face_locations = face_recognition.face_locations(frame)
#                 face_encodings = face_recognition.face_encodings(frame, face_locations)

#                 if not face_encodings:
#                     flash("No face detected in the captured image.", "danger")
#                     return redirect(url_for('attendance'))

#                 attendance_data = []
#                 # Draw bounding boxes with names and percentages
#                 for face_encoding, face_location in zip(face_encodings, face_locations):                    
#                     distances = face_recognition.face_distance(known_encodings, face_encoding)
#                     if any(distances <= 0.6):  # Threshold for recognizing faces
#                         best_match_index = np.argmin(distances)
#                         name = known_names[best_match_index]
#                         match_percentage = (1 - distances[best_match_index]) * 100  # Convert distance to percentage

#                         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                         attendance_data.append({
#                             'Timestamp': current_time,
#                             'Name': name,
#                             'Method': 'Attendance via System Camera'  # Add "Method" column
#                         })
#                         flash(f"Attendance recorded for {name} via System CV.", "success")

#                         # Draw bounding box and label
#                         top, right, bottom, left = face_location
#                         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
#                         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green bounding box
#                         cv2.putText(
#                             frame,
#                             f"{name} ({match_percentage:.2f}%)",
#                             (left, top - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.6,
#                             (0, 255, 0),
#                             2
#                         )

#                 # Save attendance to the file
#                 if attendance_data:
#                     pd.DataFrame(attendance_data).to_csv(
#                         ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE)
#                     )

#                 # Save the processed frame with bounding boxes for display
#                 processed_image_path = os.path.join(FACES_DIR, "processed_temp.jpg")
#                 cv2.imwrite(processed_image_path, frame)

#                 # Flash the success message and redirect to view attendance
#                 flash("Processing complete. Attendance recorded.", "success")

#                 return redirect(url_for('view_attendance'))
#             except Exception as e:
#                 flash(f"Error processing the captured image: {e}", "danger")
#                 return redirect(url_for('attendance'))

#         # Handle uploaded video
#         # if 'uploaded_video' in request.files and request.files['uploaded_video'].filename != '':
#         #     uploaded_file = request.files['uploaded_video']
#         #     if uploaded_file:
#         #         video_path = os.path.join(FACES_DIR, 'uploaded_video.mp4')
#         #         uploaded_file.save(video_path)

#         #         flash("Uploaded the video successfully.", "info")
#         #         cap = cv2.VideoCapture(video_path)
#         #         attendance_data = process_video_or_camera_feed(cap, known_encodings, known_names)
#         #         cap.release()

#         #         # Save attendance to the file
#         #         if attendance_data:
#         #             pd.DataFrame(attendance_data).to_csv(
#         #                 ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE)
#         #             )
#         #         flash("Processing completed. Attendance recorded.", "success")
#         #         return redirect(url_for('view_attendance'))

#         flash("No image or video was provided for attendance.", "danger")
#         return redirect(url_for('attendance'))

#     return render_template('attendance.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@socketio.on('capture_attendance')
def capture_attendance(data):
    name = data.get('name')
    if name:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_entry = {"Timestamp": current_time, "Name": name, "Method": "Attendance via System Camera"}
        # Append to CSV
        pd.DataFrame([new_entry]).to_csv(ATTENDANCE_FILE, mode='a', index=False, header=False)
        print(f"Attendance captured for {name} at {current_time}")


# Process a single frame for attendance
def process_frame(frame, known_encodings, known_names):
    attendance_data = []
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        matches = distances <= 0.6  # Threshold for recognizing faces
        if any(matches):
            best_match_index = np.argmin(distances)
            name = known_names[best_match_index]
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            attendance_data.append({'Timestamp': current_time, 'Name': name})
    return attendance_data

# Route: View Attendance
@app.route('/view-attendance')
def view_attendance():
    try:
        # Load attendance data from CSV
        if os.path.exists(ATTENDANCE_FILE):
            # Ensure proper headers while reading
            attendance_data = pd.read_csv(ATTENDANCE_FILE, header=None, names=["Timestamp", "Name", "Method"])
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
    Records attendance immediately when a face is detected.
    """
    attendance_data = []
    recorded_names = set()  # To avoid duplicate records for the same person

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

        # Detect faces and encode them
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
                name = known_names[best_match_index]

                if name not in recorded_names:  # Check if the name is already recorded
                    recorded_names.add(name)
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    attendance_data.append({'Timestamp': current_time, 'Name': name})
                    print(f"Attendance recorded for {name} ({match_percentage:.2f}%) at {current_time}")

                    # Save attendance data immediately to avoid data loss
                    pd.DataFrame(attendance_data).to_csv(
                        ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE)
                    )
                    attendance_data.clear()  # Clear the list after saving

            # Draw bounding box and name with match percentage on the face
            (top, right, bottom, left) = face_location
            color = (0, 255, 0)  # Green for valid
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                frame,
                f"{name} ({match_percentage:.2f}%)",
                (left, top - 10),  # Place text slightly above the bounding box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        # Display the frame
        # cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Video", 800, 600)
        # cv2.imshow("Video", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete. Closing video feed.")
    return attendance_data





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


