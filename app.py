import threading
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash,Response
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
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'
MJPEG_URL = os.getenv("URL_OUT")
URL_PANTRY = os.getenv("URL_PANTRY")
URL_SALES =os.getenv("URL_IN")
URL_HR = os.getenv("URL_OUT")


socketio = SocketIO(app,cors_allowed_origins="*", max_http_buffer_size=1e8, ping_timeout=120, ping_interval=25)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler()])


# Directories for storing data
FACES_DIR = 'faces'
ATTENDANCE_FILE = 'attendance.csv'
os.makedirs(FACES_DIR, exist_ok=True)
# MJPEG URL (for camera feed)
# MJPEG_URL = 'http://192.168.11.130:8080/video'



frame_interval = 1 / 10  # 10 FPS
frame_buffer = []

# Track last recorded time
last_recorded_time = {}
process_frame_interval = 3

# Load existing encodings
def load_encodings(encodings_path):
    encodings, names = [], []
    for file in os.listdir(encodings_path):
        if file.endswith("_encoding.npy"):
            name = file.split('_')[0]
            encoding = np.load(os.path.join(encodings_path, file),allow_pickle=True)
            encodings.append(encoding)
            names.append(name)
    return encodings, names

# Global variables for known encodings and names
known_encodings, known_names = load_encodings(FACES_DIR)

# Track the last recorded time for each person
last_recorded_time = defaultdict(lambda: datetime.min) 

# Stores recognized faces and their locations for continuity
recognized_faces_data = {}

# Global variables for each camera's frames and flags
current_frame_pantry = None
current_frame_sales = None
current_frame_hr = None
current_frame = None

frame_available_pantry = False
frame_available_sales = False
frame_available_hr = False
frame_available = False
frame_lock = threading.Lock()


# Example: Using a dictionary to hold the frame and availability flag for each camera
frame_data = {
    "pantry": {"frame": None, "available": False},
    "sales": {"frame": None, "available": False},
    "hr": {"frame": None, "available": False},
    "in": {"frame": None, "available": False},
    "out": {"frame": None, "available": False}
}


def process_faces(frame, known_encodings, known_names,method):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_faces = []
    now = datetime.now()
    
    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=["Name", "match percentage", "Method", "Timestamp_IN", "Timestamp_OUT"])


    for face_encoding, face_location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        match_percentage = 0

        if any(distances <= 0.5):  # Reduced threshold to reduce false positives
            best_match_index = np.argmin(distances)
            name = known_names[best_match_index]
            match_percentage = (1 - distances[best_match_index]) * 100
            recognized_faces.append({'name': name, 'match_percentage': match_percentage})
            # Check if the attendance file exists and if it's empty
            

            if name not in last_recorded_time or (now - last_recorded_time[name]).total_seconds() > 30:
                last_recorded_time[name] = now
                
                if method == 'IN':
                    attendance_data = {
                            "Timestamp_IN": now.strftime('%Y-%m-%d %H:%M:%S'),
                            "Name": name,
                            "match percentage": f'{match_percentage:.2f}',
                            "Method": method,
                            "Timestamp_OUT": '',
                            "Timestamp": now.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    df = pd.concat([df, pd.DataFrame([attendance_data])], ignore_index=True)
                    print(f"Recorded IN for {name} ({match_percentage:.2f}) at {attendance_data['Timestamp_IN']}")
                    
                elif method == 'OUT':
                    # Step 1: Filter for the most recent "IN" entry for the person
                    last_in_row = df[(df['Name'] == name)]

                    if not last_in_row.empty:
                        # Sort the rows to get the most recent "IN" entry
                        last_in_row = last_in_row.sort_values(by='Timestamp_IN', ascending=False).iloc[0]

                        # Step 2: If the most recent "IN" has an empty "Timestamp_OUT", update it
                        if pd.isna(last_in_row['Timestamp_OUT']):
                            df.loc[last_in_row.name, 'Timestamp_OUT'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(f"Updated OUT for {name} at {df.loc[last_in_row.name, 'Timestamp_OUT']}")
                        else:
                            # Step 3: If there is already a Timestamp_OUT, add a new OUT row
                            print(f"Timestamp_OUT already exists for {name}. Adding new OUT row.")
                            new_out_data = {
                                "Name": name,
                                "match percentage": f'{last_in_row["match percentage"]:.2f}',  # Keep the same match percentage from the last IN
                                "Method": 'OUT',
                                "Timestamp_IN": '',  # Empty IN for OUT
                                "Timestamp_OUT": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            df = pd.concat([df, pd.DataFrame([new_out_data])], ignore_index=True)
                            print(f"Added new OUT for {name} at {new_out_data['Timestamp_OUT']}")
                    else:
                        print(f"Timestamp_OUT already exists for {name}. Adding new OUT row.")
                        new_out_data = {
                            "Name": name,
                            "match percentage": f'{last_in_row["match percentage"]:.2f}',  # Keep the same match percentage from the last IN
                            "Method": 'OUT',
                            "Timestamp_IN": '',  # Empty IN for OUT
                            "Timestamp_OUT": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        df = pd.concat([df, pd.DataFrame([new_out_data])], ignore_index=True)
                        print(f"Added new OUT for {name} at {new_out_data['Timestamp_OUT']}")



                else:
                    attendance_data = {
                    "Timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
                    "Name": name,
                    "match percentage":f'{match_percentage:.2f}',
                    "Method": method,
                    "Timestamp_IN":'',
                    "Timestamp_OUT":''
                }
                    df = pd.concat([df, pd.DataFrame([attendance_data])], ignore_index=True)
                    print(f"Recorded Attendance for {name}({match_percentage:.2f}) at {attendance_data['Timestamp']} in {method}")


                # Append to CSV
                df.to_csv(ATTENDANCE_FILE, mode='w', index=False)   

        # Draw the bounding box and name with match percentage
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({match_percentage:.2f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, recognized_faces

# Function to capture frames in a separate thread
def capture_frames():
    global current_frame, frame_available
    capture = cv2.VideoCapture(MJPEG_URL)
    print('Thread started')
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        with frame_lock:
            current_frame = frame
            frame_available = True
        # print('captured A frame')
    capture.release()
    print('tread stopped')



def generate_frames(method):
    print('in generate')
    global current_frame, frame_available

    # Start the frame capture thread
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True  # Allow thread to exit when main program exits
    capture_thread.start()

    last_processed_time = time.time()
    
    while True:
        with frame_lock:
            if not frame_available:
                continue  # Wait for a new frame to be available
            
            # Reset the flag after capturing the current frame
            frame_available = False
            
            # Use the current frame for processing
            if current_frame is not None:
                frame_to_process = current_frame.copy()
            else:
                continue  # Skip if there is no valid frame

        current_time = time.time()
        
        # Load the latest encodings on each frame processing to handle newly added faces
        known_encodings, known_names = load_encodings(FACES_DIR)

        # Resize frame dynamically (optional based on your needs)
        resized_frame = cv2.resize(frame_to_process, (640, 480))

        # Check if enough time has passed to process the next frame
        if current_time - last_processed_time < frame_interval:
            continue
        
        last_processed_time = current_time

        # Process the frame for faces (face recognition + attendance logging)
        processed_frame, recognized_faces = process_faces(resized_frame, known_encodings, known_names,method)

        # Yield the processed frame to the client
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

def generate_frames_pantry():
    logging.info(f"{threading.current_thread().name} started - Pantry feed processing")
    last_processed_time = time.time()

    while True:
        capture_1 = cv2.VideoCapture(URL_PANTRY)
        ret_1, frame_1 = capture_1.read()
        capture_1.release()

        if ret_1:
            with frame_lock:
                frame_data["pantry"]["frame"] = frame_1
                frame_data["pantry"]["available"] = True
                known_encodings, known_names = load_encodings(FACES_DIR)
                processed_frame_1, _ = process_faces(frame_1, known_encodings, known_names,method='Pantry')

        current_time = time.time()

        if current_time - last_processed_time >= frame_interval:
            last_processed_time = current_time

            _, buffer_1 = cv2.imencode('.jpg', processed_frame_1) if processed_frame_1 is not None else (None, None)
            frame_data_1 = buffer_1.tobytes() if buffer_1 is not None else None

            # Flush the log immediately after processing
            logging.getLogger().handlers[0].flush()
        
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data_1 + b'\r\n\r\n')
        else:
            time.sleep(1)

def generate_frames_sales():
    logging.info(f"{threading.current_thread().name} started - IN feed processing")

    last_processed_time = time.time()

    while True:
        capture_2 = cv2.VideoCapture(URL_SALES)
        ret_2, frame_2 = capture_2.read()
        capture_2.release()

        if ret_2:
            with frame_lock:
                frame_data["sales"]["frame"] = frame_2
                frame_data["sales"]["available"] = True
                known_encodings, known_names = load_encodings(FACES_DIR)
                processed_frame_2, _ = process_faces(frame_2, known_encodings, known_names,method='IN')

        current_time = time.time()

        if current_time - last_processed_time >= frame_interval:
            last_processed_time = current_time

            _, buffer_2 = cv2.imencode('.jpg', processed_frame_2) if processed_frame_2 is not None else (None, None)
            frame_data_2 = buffer_2.tobytes() if buffer_2 is not None else None
            
            # Flush the log immediately after processing
            logging.getLogger().handlers[0].flush()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data_2 + b'\r\n\r\n')
        else:
            time.sleep(0.1)

# Function to capture and process frames from Camera 3
def generate_frames_hr():
    logging.info(f"{threading.current_thread().name} started - Out feed processing")

    last_processed_time = time.time()

    while True:
        capture_3 = cv2.VideoCapture(URL_HR)
        ret_3, frame_3 = capture_3.read()
        capture_3.release()

        if ret_3:
            with frame_lock:
                frame_data["hr"]["frame"] = frame_3
                frame_data["hr"]["available"] = True
                known_encodings, known_names = load_encodings(FACES_DIR)
                processed_frame_3, _ = process_faces(frame_3, known_encodings, known_names,method='OUT')

        current_time = time.time()

        if current_time - last_processed_time >= frame_interval:
            last_processed_time = current_time

            _, buffer_3 = cv2.imencode('.jpg', processed_frame_3) if processed_frame_3 is not None else (None, None)
            frame_data_3 = buffer_3.tobytes() if buffer_3 is not None else None
            
            # Flush the log immediately after processing
            logging.getLogger().handlers[0].flush()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data_3 + b'\r\n\r\n')
        else:
            time.sleep(1)

def generate_frames_test_IN():
    logging.info(f"{threading.current_thread().name} started - IN feed processing")

    capture_3 = cv2.VideoCapture(0)  # Open the camera once
    if not capture_3.isOpened():
        logging.error("Failed to open the camera")
        return

    last_processed_time = time.time()

    while True:
        ret_3, frame_3 = capture_3.read()  # Read a frame from the camera
        if not ret_3:
            logging.error("Failed to grab frame")
            break

        with frame_lock:
            frame_data["in"]["frame"] = frame_3
            frame_data["in"]["available"] = True
            known_encodings, known_names = load_encodings(FACES_DIR)
            processed_frame_3, _ = process_faces(frame_3, known_encodings, known_names, method='IN')

        current_time = time.time()

        if current_time - last_processed_time >= frame_interval:
            last_processed_time = current_time

            _, buffer_3 = cv2.imencode('.jpg', processed_frame_3) if processed_frame_3 is not None else (None, None)
            frame_data_3 = buffer_3.tobytes() if buffer_3 is not None else None
            
            # Flush the log immediately after processing
            logging.getLogger().handlers[0].flush()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data_3 + b'\r\n\r\n')

        else:
            time.sleep(0.1)  # Sleep for a small duration to limit the frame rate

    capture_3.release()  # Release the camera once the loop ends
    
def generate_frames_test_OUT():
    logging.info(f"{threading.current_thread().name} started - OUT feed processing")

    capture_3 = cv2.VideoCapture(0)  # Open the camera once
    if not capture_3.isOpened():
        logging.error("Failed to open the camera")
        return

    last_processed_time = time.time()

    while True:
        ret_3, frame_3 = capture_3.read()  # Read a frame from the camera
        if not ret_3:
            logging.error("Failed to grab frame")
            break

        with frame_lock:
            frame_data["out"]["frame"] = frame_3
            frame_data["out"]["available"] = True
            known_encodings, known_names = load_encodings(FACES_DIR)
            processed_frame_3, _ = process_faces(frame_3, known_encodings, known_names, method='OUT')

        current_time = time.time()

        if current_time - last_processed_time >= frame_interval:
            last_processed_time = current_time

            _, buffer_3 = cv2.imencode('.jpg', processed_frame_3) if processed_frame_3 is not None else (None, None)
            frame_data_3 = buffer_3.tobytes() if buffer_3 is not None else None
            
            # Flush the log immediately after processing
            logging.getLogger().handlers[0].flush()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data_3 + b'\r\n\r\n')

        else:
            time.sleep(0.1)  # Sleep for a small duration to limit the frame rate

    capture_3.release()  # Release the camera once the loop ends


def generate_frames_pantry_thread(frame_data):
    logging.info(f"{threading.current_thread().name} started - Pantry feed processing")
    capture_1 = cv2.VideoCapture(URL_PANTRY)  # Open the capture once

    if not capture_1.isOpened():
        logging.error("Failed to open camera stream for Pantry")
        return

    last_processed_time = time.time()

    while True:
        current_time = time.time()
        
        if current_time - last_processed_time >= frame_interval:
            ret_1, frame_1 = capture_1.read()

            if ret_1:
                with frame_lock:  # Use the lock when updating the shared frame data
                    # Update the shared frame data with the captured frame
                    frame_data["pantry"]["frame"] = frame_1
                    frame_data["pantry"]["available"] = True
                    known_encodings, known_names = load_encodings(FACES_DIR)
                    processed_frame_1, _ = process_faces(frame_1, known_encodings, known_names,method='Pantry')
            
            last_processed_time = current_time  # Update last processed time
        else:
            time.sleep(0.1)  # Sleep for a small period to avoid high CPU usage

    capture_1.release()  # Release the capture when done



def generate_frames_sales_thread(frame_data):
    logging.info(f"{threading.current_thread().name} started - IN feed processing")
    capture_2 = cv2.VideoCapture(URL_SALES)  # Open the capture once

    if not capture_2.isOpened():
        logging.error("Failed to open camera stream for IN")
        return

    last_processed_time = time.time()

    while True:
        current_time = time.time()
        
        if current_time - last_processed_time >= frame_interval:
            ret_2, frame_2 = capture_2.read()

            if ret_2:
                with frame_lock:  # Use the lock when updating the shared frame data
                    # Update the shared frame data with the captured frame
                    frame_data["sales"]["frame"] = frame_2
                    frame_data["sales"]["available"] = True
                    known_encodings, known_names = load_encodings(FACES_DIR)
                    processed_frame_2, _ = process_faces(frame_2, known_encodings, known_names,method='IN')
            
            last_processed_time = current_time  # Update last processed time
        else:
            time.sleep(0.1)  # Sleep for a small period to avoid high CPU usage

    capture_2.release()  # Release the capture when done


# Function to capture and process frames from Camera 3
def generate_frames_hr_thread(frame_data):
    logging.info(f"{threading.current_thread().name} started - Out feed processing")
    capture_3 = cv2.VideoCapture(URL_HR)  # Open the capture once

    if not capture_3.isOpened():
        logging.error("Failed to open camera stream for Out")
        return

    last_processed_time = time.time()

    while True:
        current_time = time.time()
        
        if current_time - last_processed_time >= frame_interval:
            ret_3, frame_3 = capture_3.read()

            if ret_3:
                with frame_lock:  # Use the lock when updating the shared frame data
                    # Update the shared frame data with the captured frame
                    frame_data["hr"]["frame"] = frame_3
                    frame_data["hr"]["available"] = True
                    known_encodings, known_names = load_encodings(FACES_DIR)
                    processed_frame_3, _ = process_faces(frame_3, known_encodings, known_names,method='Out')
            
            last_processed_time = current_time  # Update last processed time
        else:
            time.sleep(0.1)  # Sleep for a small period to avoid high CPU usage

    capture_3.release()  # Release the capture when done

            
def start_threads():
    # Pass the shared frame_data object to each thread
    # pantry_thread = threading.Thread(target=generate_frames_pantry_thread, args=(frame_data,), daemon=True)
    # pantry_thread.start()
    # logging.info(f"{pantry_thread.name} started")

    sales_thread = threading.Thread(target=generate_frames_sales_thread, args=(frame_data,), daemon=True)
    sales_thread.start()
    logging.info(f"{sales_thread.name} started")

    hr_thread = threading.Thread(target=generate_frames_hr_thread, args=(frame_data,), daemon=True)
    hr_thread.start()
    logging.info(f"{hr_thread.name} started")

@app.route('/clear-attendance', methods=['GET'])
def clear_attendance():
    try:
        # Check if the attendance file exists
        if os.path.exists(ATTENDANCE_FILE):
            # Open the file in read and write mode
            with open(ATTENDANCE_FILE, 'r+') as file:
                lines = file.readlines()  # Read all lines from the file
                if len(lines) > 1:
                    # Keep the header (first line) and truncate the file after the header
                    header = lines[0]
                    file.seek(0)  # Move the pointer to the beginning of the file
                    file.write(header)  # Write the header back to the file
                    file.truncate()  # Clear everything after the header

            return jsonify({"message": "Attendance data cleared successfully, header preserved!"}), 200
        else:
            return jsonify({"message": "Attendance file does not exist!"}), 404
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"message": f"Error: {str(e)}"}), 500




@app.route('/video-feed-pantry')
def video_feed_pantry():
    return Response(generate_frames_pantry(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video-feed-in')
def video_feed_sales():
    # return Response(generate_frames_sales(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_test_IN(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video-feed-out')
def video_feed_hr():
    # return Response(generate_frames_hr(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_test_OUT(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/video-feed')
def video_feed():
    # return render_template('all_frames.html')
    return Response(generate_frames('test'), mimetype='multipart/x-mixed-replace; boundary=frame')

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

# Route: Attendance
@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

# Route: Home
@app.route('/')
def index():
    return render_template('index.html')

# Route: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        captured_image = request.form['captured_image']

        # Decode the base64 image data
        img_data = captured_image.split(",")[1]
        img_binary = base64.b64decode(img_data)

        # Save the image to the 'faces' directory
        image_path = os.path.join(FACES_DIR, f"{name}.jpg")
        with open(image_path, "wb") as f:
            f.write(img_binary)

        # Load the image to get the face encoding
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the face encodings for the image
        face_encodings = face_recognition.face_encodings(rgb_image)

        if face_encodings:
            encoding_path = os.path.join(FACES_DIR, f"{name}_encoding.npy")
            np.save(encoding_path, face_encodings[0])  # Save the first face encoding
            return render_template('register.html', success=f"Face for {name} registered successfully!")
        else:
            return render_template('register.html', error="No face detected. Please try again.")
    
    return render_template('register.html')



# Route: View Attendance
@app.route('/view-attendance')
def view_attendance():
    try:
        # Load attendance data from CSV
        if os.path.exists(ATTENDANCE_FILE):
            attendance_data = pd.read_csv(ATTENDANCE_FILE, header=None, names=["Name","Percentage", "Method","Timestamp_IN","Timestamp_OUT","Timestamp"])
            records = attendance_data.to_dict(orient='records')  # Convert to list of dictionaries
        else:
            records = []  # Empty if the file doesn't exist
    except Exception as e:
        print(f"Error loading attendance: {e}")
        records = []

    return render_template('view_attendance.html', records=records)



if __name__ == '__main__':

    # start_threads()
    # generate_frames()
    socketio.run(app, debug=False,allow_unsafe_werkzeug=True)
