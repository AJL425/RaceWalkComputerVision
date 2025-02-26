import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directory paths
input_folder = "dataset/raw_images"
output_folder = "dataset/annotated_images"

os.makedirs(output_folder, exist_ok=True)

# Define foot landmark indices
LEFT_FOOT = [27, 29, 31]
RIGHT_FOOT = [28, 32, 30]

# Process images
for filename in os.listdir(input_folder):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw foot landmarks on image
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)

            if idx in LEFT_FOOT or idx in RIGHT_FOOT:
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Green dot

        # Save the new image with landmarks
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")
