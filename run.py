import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- 1. INITIAL SETUP ---

# --- Configuration ---
class Config:
    # Screen dimensions
    SCREEN_W, SCREEN_H = pyautogui.size()

    # Smoothing
    SMOOTH_FACTOR = 0.8

    # Region of Interest (ROI) for eye movement sensitivity
    # Normalized from 0.0 to 1.0 (relative to the face/frame)
    X_MIN, X_MAX = 0.3, 0.7  # Horizontal sensitivity
    Y_MIN, Y_MAX = 0.35, 0.65 # Vertical sensitivity

# --- Landmark Indices (from MediaPipe) ---
LEFT_EYE_PUPIL_INDICES = [468, 469, 470, 471, 472] # Using refined iris landmarks
RIGHT_EYE_PUPIL_INDICES = [473, 474, 475, 476, 477]

def calculate_eye_center(landmarks, indices, img_w, img_h, image_for_drawing=None):
    """Calculates the center of an eye given its landmarks."""
    x_sum, y_sum = 0, 0
    for index in indices:
        lm = landmarks[index]
        x_sum += lm.x
        y_sum += lm.y
        if image_for_drawing is not None:
            cv2.circle(image_for_drawing, (int(lm.x * img_w), int(lm.y * img_h)), 2, (0, 255, 0), -1)
    
    norm_x = x_sum / len(indices)
    norm_y = y_sum / len(indices)
    return norm_x, norm_y

print("Starting Eye Tracking. Press 'q' to exit.")

# --- 2. INITIALIZE COMPONENTS ---

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize tracking variables
current_gaze_x = Config.SCREEN_W // 2
current_gaze_y = Config.SCREEN_H // 2

# --- 2. TRACKING LOOP ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    # Mirror the image (flip=1) for a more intuitive display
    image = cv2.flip(image, 1) 
    
    # Convert the image to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect the face mesh
    results = face_mesh.process(image_rgb)
    
    img_h, img_w, _ = image.shape
    
    # --- 3. GAZE CALCULATION AND CURSOR CONTROL ---
    if results.multi_face_landmarks:
        
        # We only consider the first detected face
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # --- Calculate Eye Centers ---
        norm_left_pupil_x, norm_left_pupil_y = calculate_eye_center(face_landmarks, LEFT_EYE_PUPIL_INDICES, img_w, img_h, image)
        norm_right_pupil_x, norm_right_pupil_y = calculate_eye_center(face_landmarks, RIGHT_EYE_PUPIL_INDICES, img_w, img_h, image)

        # --- Average Both Eyes for a Stable Gaze Point ---
        norm_pupil_x = (norm_left_pupil_x + norm_right_pupil_x) / 2.0
        norm_pupil_y = (norm_left_pupil_y + norm_right_pupil_y) / 2.0

        # --- Map Gaze to Screen Coordinates ---
        gaze_x_map = np.clip((norm_pupil_x - Config.X_MIN) / (Config.X_MAX - Config.X_MIN), 0.0, 1.0)
        gaze_y_map = np.clip((norm_pupil_y - Config.Y_MIN) / (Config.Y_MAX - Config.Y_MIN), 0.0, 1.0)
        
        target_cursor_x = int(gaze_x_map * Config.SCREEN_W)
        target_cursor_y = int(gaze_y_map * Config.SCREEN_H)

        # --- Smooth Cursor Movement ---
        current_gaze_x = int(current_gaze_x * Config.SMOOTH_FACTOR + target_cursor_x * (1.0 - Config.SMOOTH_FACTOR))
        current_gaze_y = int(current_gaze_y * Config.SMOOTH_FACTOR + target_cursor_y * (1.0 - Config.SMOOTH_FACTOR))

        pyautogui.moveTo(current_gaze_x, current_gaze_y)
        # Log the current cursor coordinates to the console
        print(f"Cursor Position: ({current_gaze_x}, {current_gaze_y})")
        
        
        # Optional: Display the mapped gaze point on the webcam feed
        # cv2.circle(image, (int(norm_pupil_x * img_w), int(norm_pupil_y * img_h)), 5, (0, 0, 255), -1)
        # cv2.putText(image, "GAZE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # --- 4. DISPLAY AND EXIT ---
    
    # Display the processed image
    # cv2.imshow('Eye Controlled Cursor', image)

    # Exit on 'q' press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()