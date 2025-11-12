import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from dotenv import load_dotenv
import os

load_dotenv()

# --- 1. SET YOUR MODE AND FILE PATH ---
# Switch to 'webcam' to test your camera
MODE = 'webcam' 

# Make sure this path is correct for video mode
VIDEO_FILE_PATH = os.getenv("VIDEO_FILE_PATH") 

# --- 2. SET TUNING PARAMETERS ---
EAR_THRESHOLD = 0.22         
CONSECUTIVE_FRAMES = 3     

# --- 3. MEDIAPIPE LANDMARK CONSTANTS ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Iris landmarks (Top, Bottom)
RIGHT_IRIS_INDICES = [469, 471] 
LEFT_IRIS_INDICES = [474, 476]

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

    cap = None # Initialize cap to None
    
    if MODE == 'webcam':
        print("Searching for webcam...")
        # --- NEW: Test camera indices 0, 1, and 2 ---
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Webcam found at index {i}")
                source = i
                break
        if not cap or not cap.isOpened():
            print(f"Error: Could not find or open any webcam.")
            return
    
    elif MODE == 'video':
        source = VIDEO_FILE_PATH
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {source}")
            print("Please check the VIDEO_FILE_PATH variable.")
            return

    # --- 2. SET UP CSV LOGGING ---
    csv_file_name = 'blink_data_log_enhanced.csv'
    csv_header = [
        'timestamp_sec', 'avg_ear', 'left_ear', 'right_ear', 'is_blinking_frame', 
        'closed_frame_counter', 'blink_count_total', 'blink_rate_bps',
        'left_pupil_diameter', 'right_pupil_diameter'
    ]
    
    try:
        log_file = open(csv_file_name, 'w', newline='')
        writer = csv.writer(log_file)
        writer.writerow(csv_header)
    except IOError as e:
        print(f"Error: Could not open CSV file for writing: {e}")
        cap.release() # Release camera/video
        return

    # --- 3. SET UP COUNTERS AND TIMERS ---
    blink_counter = 0
    closed_frame_counter = 0
    start_time = time.time()
    blinks_per_second = 0.0
    elapsed_time = 0.0

    print("Starting video processing... Press 'q' to quit.")

    # --- 4. START PROCESSING LOOP ---
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                if MODE == 'video': print("End of video file.")
                else: print("Camera feed lost.")
                break

            if MODE == 'webcam':
                frame = cv2.flip(frame, 1)

            img_h, img_w, _ = frame.shape
            
            # --- 5. MEDIAPIPE & BLINK LOGIC ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Reset all metrics for the current frame
            avg_ear = 0.0; left_ear = 0.0; right_ear = 0.0
            left_pupil_diameter = 0.0; right_pupil_diameter = 0.0
            is_blinking_now = 0

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                
                # --- NEW: ROBUSTNESS CHECK ---
                # Check if the detailed 478 landmarks (including iris) are present
                # If eyes are closed, this will be False
                has_iris_landmarks = len(landmarks) == 478 
                
                # --- Calculate EAR (always possible) ---
                right_eye_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE_INDICES]
                left_eye_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE_INDICES]
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # --- Safely Calculate Pupil Diameter ---
                if has_iris_landmarks:
                    try:
                        right_iris_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_IRIS_INDICES]
                        left_iris_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_IRIS_INDICES]
                    
                        left_pupil_diameter = calculate_pixel_distance(left_iris_points[0], left_iris_points[1])
                        right_pupil_diameter = calculate_pixel_distance(right_iris_points[0], right_iris_points[1])
                    except Exception as e:
                        # This is a fallback, but the length check should prevent it
                        print(f"Error calculating pupil diameter: {e}")
                        left_pupil_diameter = 0.0
                        right_pupil_diameter = 0.0
                
                # --- Blink Counting Logic ---
                if avg_ear < EAR_THRESHOLD:
                    closed_frame_counter += 1
                    is_blinking_now = 1
                else:
                    if closed_frame_counter >= CONSECUTIVE_FRAMES:
                        blink_counter += 1
                    closed_frame_counter = 0 # Reset counter when eye is open
            
            # --- 6. CALCULATE ROLLING BLINK RATE ---
            elapsed_time = time.time() - start_time
            if elapsed_time > 1: blinks_per_second = blink_counter / elapsed_time
            else: blinks_per_second = 0.0

            # --- 7. WRITE ENHANCED DATA TO CSV ---
            current_data_row = [
                f"{elapsed_time:.3f}", f"{avg_ear:.4f}", f"{left_ear:.4f}", f"{right_ear:.4f}",
                is_blinking_now, closed_frame_counter, blink_counter, f"{blinks_per_second:.4f}",
                f"{left_pupil_diameter:.4f}", f"{right_pupil_diameter:.4f}"
            ]
            writer.writerow(current_data_row)

            # --- 8. DISPLAY RESULTS ---
            cv2.putText(frame, f"BLINKS: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink Rate (B/s): {blinks_per_second:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Blink Detection Data Collection', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # --- 9. CLEANUP ---
        if log_file:
            log_file.close()
            print(f"--- Data Collection Complete ---")
            print(f"Data saved to: {csv_file_name}")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        
        if elapsed_time > 0:
            print(f"Total Blinks: {blink_counter}")
            print(f"Total Duration: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()