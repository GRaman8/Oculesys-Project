import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION ---
# Switch to 'webcam' or 'video'
MODE = 'video' 

# Folder containing your videos (e.g., "C:/Users/Name/Videos/TestSet")
# Use forward slashes (/) or double backslashes (\\)
VIDEO_FOLDER_PATH = os.getenv("VIDEO_FOLDER_PATH") 

# Output directory for CSV files
LOG_DIRECTORY = "timestamps"

# --- 2. TUNING PARAMETERS ---
EAR_THRESHOLD = 0.22         
CONSECUTIVE_FRAMES = 3     

# --- 3. LANDMARK CONSTANTS ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_IRIS_INDICES = [469, 471]
LEFT_IRIS_INDICES = [474, 476]

# --- 4. 3D MODEL POINTS FOR HEAD POSE ---
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

def calculate_ear(eye_points):
    try:
        p1=np.array(eye_points[0]); p2=np.array(eye_points[1]); p3=np.array(eye_points[2])
        p4=np.array(eye_points[3]); p5=np.array(eye_points[4]); p6=np.array(eye_points[5])
        dist_p2_p6=np.linalg.norm(p2-p6); dist_p3_p5=np.linalg.norm(p3-p5)
        dist_p1_p4=np.linalg.norm(p1-p4)
        if dist_p1_p4 == 0: return 0.0
        return (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)
    except: return 0.0

def calculate_pixel_distance(point1, point2):
    try:
        return np.linalg.norm(np.array(point1) - np.array(point2))
    except: return 0.0

def process_single_source(source_path, is_webcam, face_mesh):
    """
    Handles the processing logic for a SINGLE video or webcam session.
    """
    cap = None
    csv_file_path = ""
    
    # --- SETUP SOURCE AND CSV NAME ---
    if is_webcam:
        print("Searching for webcam...")
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Webcam found at index {i}")
                break
        if not cap or not cap.isOpened():
            print("Error: Could not find webcam.")
            return

        # Webcam CSV Naming
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        counter = 1
        base_name = f"webcam_take_{counter}.csv"
        while os.path.exists(os.path.join(LOG_DIRECTORY, base_name)):
            counter += 1
            base_name = f"webcam_take_{counter}.csv"
        csv_file_path = os.path.join(LOG_DIRECTORY, base_name)
        
        # Webcam Timer
        try:
            duration = float(input("Enter duration in minutes (e.g., 0.5): "))
            end_time = time.time() + (duration * 60)
        except:
            end_time = time.time() + 60
            print("Invalid input. Defaulting to 1 minute.")
            
    else: # Video File Mode
        print(f"--- Processing: {os.path.basename(source_path)} ---")
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error opening file: {source_path}")
            return

        # Video CSV Naming (based on filename)
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(source_path))[0]
        csv_file_path = os.path.join(LOG_DIRECTORY, f"{video_name}_log.csv")
        end_time = float('inf')

    print(f"Saving data to: {csv_file_path}")

    # --- CSV SETUP ---
    csv_header = [
        'timestamp_sec', 'avg_ear', 'left_ear', 'right_ear', 'is_blinking_frame', 
        'closed_frame_counter', 'blink_count_total', 'blink_rate_bps',
        'left_pupil_diameter', 'right_pupil_diameter',
        'head_pitch', 'head_yaw', 'head_roll', 'nose_z'
    ]
    
    try:
        log_file = open(csv_file_path, 'w', newline='')
        writer = csv.writer(log_file)
        writer.writerow(csv_header)
    except IOError as e:
        print(f"Error opening CSV: {e}")
        cap.release()
        return

    # --- INIT VARIABLES ---
    blink_counter = 0
    closed_frame_counter = 0
    start_time = time.time()
    blinks_per_second = 0.0
    
    camera_matrix = None
    dist_coeffs = np.zeros((4,1))

    # --- PROCESSING LOOP ---
    try:
        while cap.isOpened():
            if time.time() > end_time:
                print("Duration complete.")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Skipping/Stopping current source...")
                break

            success, frame = cap.read()
            if not success:
                break # End of video

            if is_webcam:
                frame = cv2.flip(frame, 1)

            img_h, img_w, _ = frame.shape
            
            # Initialize Camera Matrix once
            if camera_matrix is None:
                focal_length = img_w
                center = (img_w / 2, img_h / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]], dtype=np.float64)

            # Run MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # Reset frame metrics
            avg_ear = 0.0; left_ear = 0.0; right_ear = 0.0
            left_pupil_diameter = 0.0; right_pupil_diameter = 0.0
            is_blinking_now = 0
            head_pitch = 0.0; head_yaw = 0.0; head_roll = 0.0; nose_z = 0.0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                has_iris = len(landmarks) == 478

                # 1. Head Pose
                try:
                    image_points = np.array([
                        (landmarks[1].x * img_w, landmarks[1].y * img_h),
                        (landmarks[152].x * img_w, landmarks[152].y * img_h),
                        (landmarks[263].x * img_w, landmarks[263].y * img_h),
                        (landmarks[33].x * img_w, landmarks[33].y * img_h),
                        (landmarks[287].x * img_w, landmarks[287].y * img_h),
                        (landmarks[57].x * img_w, landmarks[57].y * img_h)
                    ], dtype=np.float64)
                    
                    _, rot_vec, trans_vec = cv2.solvePnP(
                        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    rot_mat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                    head_pitch, head_yaw, head_roll = angles[0], angles[1], angles[2]
                    nose_z = landmarks[1].z
                except: pass

                # 2. EAR & Blink
                r_pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE_INDICES]
                l_pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE_INDICES]
                left_ear = calculate_ear(l_pts)
                right_ear = calculate_ear(r_pts)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    closed_frame_counter += 1
                    is_blinking_now = 1
                else:
                    if closed_frame_counter >= CONSECUTIVE_FRAMES:
                        blink_counter += 1
                    closed_frame_counter = 0

                # 3. Pupil Diameter
                if has_iris:
                    try:
                        r_iris = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_IRIS_INDICES]
                        l_iris = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_IRIS_INDICES]
                        left_pupil_diameter = calculate_pixel_distance(l_iris[0], l_iris[1])
                        right_pupil_diameter = calculate_pixel_distance(r_iris[0], r_iris[1])
                    except: pass

            # Calculate Rate & Write CSV
            elapsed = time.time() - start_time
            blinks_per_second = blink_counter / elapsed if elapsed > 1 else 0.0
            
            writer.writerow([
                f"{elapsed:.3f}", f"{avg_ear:.4f}", f"{left_ear:.4f}", f"{right_ear:.4f}",
                is_blinking_now, closed_frame_counter, blink_counter, f"{blinks_per_second:.4f}",
                f"{left_pupil_diameter:.4f}", f"{right_pupil_diameter:.4f}",
                f"{head_pitch:.2f}", f"{head_yaw:.2f}", f"{head_roll:.2f}", f"{nose_z:.4f}"
            ])

            # Display
            cv2.putText(frame, f"BLINKS: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if is_webcam:
                rem = max(0, end_time - time.time())
                cv2.putText(frame, f"Time: {int(rem)}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"File: {os.path.basename(source_path)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow('Blink Detection', frame)

    except Exception as e:
        print(f"Error processing {source_path}: {e}")
    finally:
        log_file.close()
        cap.release()
        if elapsed > 0:
            print(f"  -> Total Blinks: {blink_counter}")
            print(f"  -> Duration: {elapsed:.2f}s")

def main():
    # Init MediaPipe once
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    if MODE == 'webcam':
        process_single_source(None, True, face_mesh)
    
    elif MODE == 'video':
        # --- NEW: Folder Iteration Logic ---
        if not os.path.exists(VIDEO_FOLDER_PATH):
            print(f"Error: Folder not found: {VIDEO_FOLDER_PATH}")
            return

        # Get all valid video files
        valid_exts = ('.mp4', '.mov', '.avi', '.mkv')
        files = [f for f in os.listdir(VIDEO_FOLDER_PATH) if f.lower().endswith(valid_exts)]
        
        if not files:
            print(f"No video files found in {VIDEO_FOLDER_PATH}")
            return

        print(f"Found {len(files)} videos. Starting batch processing...")
        
        for filename in files:
            full_path = os.path.join(VIDEO_FOLDER_PATH, filename)
            process_single_source(full_path, False, face_mesh)
            
        print("\n--- Batch Processing Complete ---")

    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()