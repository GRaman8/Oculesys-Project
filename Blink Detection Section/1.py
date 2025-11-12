import cv2
import mediapipe as mp
import numpy as np
import time
import csv

# --- 1. SET YOUR MODE AND FILE PATH ---
# Switch between 'webcam' or 'video'
MODE = 'video' 

# Ignored if MODE is 'webcam', otherwise set your video path
VIDEO_FILE_PATH = "/home/Fall 2025/8945_ARL/Oculesys Project/Blink Detection Section/Data/Adobe Express - Sample Data-3.mp4" 

# --- 2. SET TUNING PARAMETERS ---
# Calibrate these values by testing your videos
EAR_THRESHOLD = 0.22         
CONSECUTIVE_FRAMES = 3     

# --- 3. MEDIAPIPE LANDMARK CONSTANTS ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_points):
    """Calculates the Eye Aspect Ratio (EAR) given 6 eye landmark points."""
    try:
        p1 = np.array(eye_points[0])
        p2 = np.array(eye_points[1])
        p3 = np.array(eye_points[2])
        p4 = np.array(eye_points[3])
        p5 = np.array(eye_points[4])
        p6 = np.array(eye_points[5])
        
        dist_p2_p6 = np.linalg.norm(p2 - p6)
        dist_p3_p5 = np.linalg.norm(p3 - p5)
        dist_p1_p4 = np.linalg.norm(p1 - p4)
        
        # Avoid division by zero
        if dist_p1_p4 == 0:
            return 0.0
            
        ear = (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)
        return ear
    except Exception as e:
        print(f"Error in EAR calculation: {e}")
        return 0.0

def main():
    # --- 1. INITIALIZE MEDIAPIPE & VIDEO CAPTURE ---
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    if MODE == 'webcam':
        source = 0  # 0 is the default webcam
    elif MODE == 'video':
        source = VIDEO_FILE_PATH
    else:
        print(f"Invalid MODE: {MODE}. Please choose 'webcam' or 'video'.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    # --- 2. SET UP CSV LOGGING ---
    csv_file_name = 'blink_data_log.csv'
    # This header is ready for your time-series analysis
    csv_header = ['timestamp_sec', 'avg_ear', 'is_blinking_frame', 'blink_count_total', 'blink_rate_bps']
    
    try:
        log_file = open(csv_file_name, 'w', newline='')
        writer = csv.writer(log_file)
        writer.writerow(csv_header) # Write the header row
    except IOError as e:
        print(f"Error: Could not open CSV file for writing: {e}")
        return

    # --- 3. SET UP COUNTERS AND TIMERS ---
    blink_counter = 0
    closed_frame_counter = 0
    start_time = time.time()
    blinks_per_second = 0.0

    print("Starting video processing... Press 'q' to quit.")

    # --- 4. START PROCESSING LOOP ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            if MODE == 'video':
                print("End of video file.")
            else:
                print("Camera feed lost.")
            break

        if MODE == 'webcam':
            frame = cv2.flip(frame, 1)

        img_h, img_w, _ = frame.shape
        
        # --- 5. MEDIAPIPE & BLINK LOGIC ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # Optimize performance
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        avg_ear = 0.0
        is_blinking_now = 0 # 0 for 'no', 1 for 'yes'

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            right_eye_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE_INDICES]
            left_eye_points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE_INDICES]
            
            avg_ear = (calculate_ear(left_eye_points) + calculate_ear(right_eye_points)) / 2.0
            
            if avg_ear < EAR_THRESHOLD:
                closed_frame_counter += 1
                is_blinking_now = 1 # Mark this frame as part of a blink
            else:
                if closed_frame_counter >= CONSECUTIVE_FRAMES:
                    blink_counter += 1
                closed_frame_counter = 0
        
        # --- 6. CALCULATE ROLLING BLINK RATE ---
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 1: # Avoid division by zero
            blinks_per_second = blink_counter / elapsed_time
        else:
            blinks_per_second = 0.0

        # --- 7. WRITE DATA TO CSV FOR THIS FRAME ---
        try:
            current_data_row = [
                f"{elapsed_time:.3f}", 
                f"{avg_ear:.4f}", 
                is_blinking_now, 
                blink_counter, 
                f"{blinks_per_second:.4f}"
            ]
            writer.writerow(current_data_row)
        except IOError:
            print("Error: Could not write to CSV file.")

        # --- 8. DISPLAY RESULTS (FOR LOCAL MACHINE) ---
        cv2.putText(frame, f"BLINKS: {blink_counter}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blink Rate (B/s): {blinks_per_second:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Blink Detection Data Collection', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- 9. CLEANUP ---
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    print(f"--- Data Collection Complete ---")
    print(f"Total Blinks: {blink_counter}")
    print(f"Total Duration: {elapsed_time:.2f} seconds")
    print(f"Average Blink Rate (B/s): {blinks_per_second:.2f}")
    print(f"Data saved to: {csv_file_name}")

if __name__ == "__main__":
    main()