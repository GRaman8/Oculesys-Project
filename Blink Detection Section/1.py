import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. SET YOUR MODE AND FILE PATH ---
# Switch to 'webcam' or 'video'
MODE = 'webcam' 

# Make sure this path is correct for video mode
VIDEO_FILE_PATH = os.getenv("VIDEO_FILE_PATH") 

# --- SET LOG DIRECTORY ---
LOG_DIRECTORY = "timestamps"

# --- 2. SET TUNING PARAMETERS ---
EAR_THRESHOLD = 0.22         
CONSECUTIVE_FRAMES = 3     

# --- 3. MEDIAPIPE LANDMARK CONSTANTS ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_IRIS_INDICES = [469, 471] # (Top, Bottom)
LEFT_IRIS_INDICES = [474, 476]  # (Top, Bottom)

def calculate_ear(eye_points):
    """Calculates the Eye Aspect Ratio (EAR) given 6 eye landmark points."""
    try:
        p1=np.array(eye_points[0]); p2=np.array(eye_points[1]); p3=np.array(eye_points[2])
        p4=np.array(eye_points[3]); p5=np.array(eye_points[4]); p6=np.array(eye_points[5])
        dist_p2_p6=np.linalg.norm(p2-p6); dist_p3_p5=np.linalg.norm(p3-p5)
        dist_p1_p4=np.linalg.norm(p1-p4)
        if dist_p1_p4 == 0: return 0.0
        ear = (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)
        return ear
    except: return 0.0

def calculate_pixel_distance(point1, point2):
    """Calculates the Euclidean distance in pixels between two (x, y) points."""
    try:
        return np.linalg.norm(np.array(point1) - np.array(point2))
    except: return 0.0

def main():
    # --- 1. INITIALIZE MEDIAPIPE & VIDEO CAPTURE ---
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True, # Enable iris detection
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = None 
    end_time = float('inf') 
    
    if MODE == 'webcam':
        print("Searching for webcam...")
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Webcam found at index {i}")
                break
        if not cap or not cap.isOpened():
            print(f"Error: Could not find or open any webcam.")
            return
        
        try:
            duration_min = float(input("Enter run duration in minutes (e.g., 0.5 for 30 seconds): "))
            end_time = time.time() + (duration_min * 60)
            print(f"Webcam will run for {duration_min} minutes.")
        except ValueError:
            print("Invalid input. Defaulting to 1 minute.")
            end_time = time.time() + 60
        print("Click on the 'Blink Detection' window and press 'q' to stop early.")
    
    elif MODE == 'video':
        source = VIDEO_FILE_PATH
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {source}")
            print("Please check the VIDEO_FILE_PATH variable.")
            return
        print("Processing video... Press 'q' in the window to stop.")

    # --- 2. SET UP DYNAMIC CSV LOGGING ---
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    
    csv_file_name = ''
    if MODE == 'video':
        base_name = os.path.basename(VIDEO_FILE_PATH)
        file_name_without_ext = os.path.splitext(base_name)[0]
        csv_file_name = f"{file_name_without_ext}_log.csv"
        
    elif MODE == 'webcam':
        counter = 1
        csv_file_name = f"webcam_take_{counter}.csv"

        while os.path.exists(os.path.join(LOG_DIRECTORY, csv_file_name)):
            counter += 1
            csv_file_name = f"webcam_take_{counter}.csv"
            
    csv_path = os.path.join(LOG_DIRECTORY, csv_file_name)
    print(f"Data will be saved to: {csv_path}")

    # --- NEW: EXPANDED CSV HEADER ---
    csv_header = [
        'timestamp_sec', 'avg_ear', 'left_ear', 'right_ear', 'is_blinking_frame', 
        'closed_frame_counter', 'blink_count_total', 'blink_rate_bps',
        'left_pupil_diameter', 'right_pupil_diameter',
        'head_pitch', 'head_yaw', 'head_roll', 'nose_z' # <-- NEW ATTRIBUTES
    ]
    
    try:
        log_file = open(csv_path, 'w', newline='')
        writer = csv.writer(log_file)
        writer.writerow(csv_header)
    except IOError as e:
        print(f"Error: Could not open CSV file for writing: {e}")
        cap.release()
        return

    # --- 3. SET UP COUNTERS, TIMERS, AND POSE MODEL ---
    blink_counter = 0
    closed_frame_counter = 0
    start_time = time.time()
    blinks_per_second = 0.0
    elapsed_time = 0.0

    # --- NEW: 3D Model and Camera setup for Head Pose ---
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
        (0.0, -330.0, -65.0),        # Chin (Landmark 152)
        (-225.0, 170.0, -135.0),     # Left eye left corner (Landmark 263)
        (225.0, 170.0, -135.0),      # Right eye right corner (Landmark 33)
        (-150.0, -150.0, -125.0),    # Left Mouth corner (Landmark 287)
        (150.0, -150.0, -125.0)      # Right mouth corner (Landmark 57)
    ], dtype=np.float64)
    
    camera_matrix = None # Will be set on the first frame
    dist_coeffs = np.zeros((4,1)) # Assume no lens distortion

    # --- 4. START PROCESSING LOOP ---
    try:
        while cap.isOpened():
            if time.time() > end_time:
                print("Webcam duration complete.")
                break
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("'q' key pressed. Stopping.")
                break

            success, frame = cap.read()
            if not success:
                if MODE == 'video': print("End of video file.")
                else: print("Camera feed lost.")
                break

            if MODE == 'webcam':
                frame = cv2.flip(frame, 1)

            img_h, img_w, _ = frame.shape
            
            # --- NEW: Initialize Camera Matrix (once) ---
            if camera_matrix is None:
                focal_length = img_w
                center = (img_w / 2, img_h / 2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype=np.float64
                )

            # --- 5. MEDIAPIPE & BLINK LOGIC ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Reset metrics
            avg_ear = 0.0; left_ear = 0.0; right_ear = 0.0
            left_pupil_diameter = 0.0; right_pupil_diameter = 0.0
            is_blinking_now = 0
            head_pitch = 0.0; head_yaw = 0.0; head_roll = 0.0; nose_z = 0.0 # <-- NEW

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                has_iris_landmarks = len(landmarks) == 478 
                
                # --- NEW: Head Pose Estimation ---
                try:
                    # Get 2D image points
                    image_points = np.array([
                        (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
                        (landmarks[152].x * img_w, landmarks[152].y * img_h),   # Chin
                        (landmarks[263].x * img_w, landmarks[263].y * img_h),   # Left eye left corner
                        (landmarks[33].x * img_w, landmarks[33].y * img_h),    # Right eye right corner
                        (landmarks[287].x * img_w, landmarks[287].y * img_h),   # Left Mouth corner
                        (landmarks[57].x * img_w, landmarks[57].y * img_h)     # Right mouth corner
                    ], dtype=np.float64)
                    
                    # Solve PnP
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(
                        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    # Convert rotation vector to angles (Pitch, Yaw, Roll)
                    (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)
                    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
                    head_pitch = angles[0]
                    head_yaw = angles[1]
                    head_roll = angles[2]
                    
                    # Get nose z-coordinate (from 3D landmarks)
                    nose_z = landmarks[1].z
                    
                except Exception as e:
                    pass # Will fail if landmarks aren't detected
                # --- End of Head Pose ---

                # --- EAR Calculation ---
                right_eye_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE_INDICES]
                left_eye_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE_INDICES]
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # --- Pupil Diameter Calculation ---
                if has_iris_landmarks:
                    try:
                        right_iris_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_IRIS_INDICES]
                        left_iris_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_IRIS_INDICES]
                        left_pupil_diameter = calculate_pixel_distance(left_iris_points[0], left_iris_points[1])
                        right_pupil_diameter = calculate_pixel_distance(right_iris_points[0], right_iris_points[1])
                    except Exception as e:
                        left_pupil_diameter = 0.0; right_pupil_diameter = 0.0
                
                # --- Blink Counting ---
                if avg_ear < EAR_THRESHOLD:
                    closed_frame_counter += 1
                    is_blinking_now = 1
                else:
                    if closed_frame_counter >= CONSECUTIVE_FRAMES:
                        blink_counter += 1
                    closed_frame_counter = 0 
            
            # --- 6. CALCULATE ROLLING BLINK RATE ---
            elapsed_time = time.time() - start_time
            if elapsed_time > 1: blinks_per_second = blink_counter / elapsed_time
            else: blinks_per_second = 0.0

            # --- 7. WRITE ENHANCED DATA TO CSV ---
            # --- NEW: Expanded Data Row ---
            current_data_row = [
                f"{elapsed_time:.3f}", f"{avg_ear:.4f}", f"{left_ear:.4f}", f"{right_ear:.4f}",
                is_blinking_now, closed_frame_counter, blink_counter, f"{blinks_per_second:.4f}",
                f"{left_pupil_diameter:.4f}", f"{right_pupil_diameter:.4f}",
                f"{head_pitch:.2f}", f"{head_yaw:.2f}", f"{head_roll:.2f}", f"{nose_z:.4f}" # <-- NEW
            ]
            writer.writerow(current_data_row)

            # --- 8. DISPLAY RESULTS ---
            cv2.putText(frame, f"BLINKS: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink Rate (B/s): {blinks_per_second:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if MODE == 'webcam':
                remaining_time = max(0, end_time - time.time())
                cv2.putText(frame, f"Time Left: {int(remaining_time)}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- NEW: Display Head Pose ---
            cv2.putText(frame, f"Pitch: {head_pitch:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Yaw: {head_yaw:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Roll: {head_roll:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Blink Detection Data Collection', frame)
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # --- 9. CLEANUP ---
        if 'log_file' in locals() and not log_file.closed:
            log_file.close()
            print(f"--- Data Collection Complete ---")
            print(f"Data saved to: {csv_path}")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        
        if elapsed_time > 0:
            print(f"Total Blinks: {blink_counter}")
            print(f"Total Duration: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()