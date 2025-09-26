import os
import cv2
import xgboost as xgb
import numpy as np
import mediapipe as mp
import json

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the directory where we store frames in SageMaker
frame_directory = 'dataset/C0834_Walker1/' 
output_directory = 'dataset/annotated/C0834_Walker1/'  

# Path to the trained model
model_path = '/path/to/your/xgboost-model.json' 
model = xgb.Booster()
model.load_model(model_path)

# Define the correct feature names as expected by the model
expected_columns = [
    'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'LEFT_ANKLE_z', 'LEFT_ANKLE_visibility',
    'LEFT_HEEL_x', 'LEFT_HEEL_y', 'LEFT_HEEL_z', 'LEFT_HEEL_visibility',
    'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y', 'LEFT_FOOT_INDEX_z', 'LEFT_FOOT_INDEX_visibility',
    'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'RIGHT_ANKLE_z', 'RIGHT_ANKLE_visibility',
    'RIGHT_HEEL_x', 'RIGHT_HEEL_y', 'RIGHT_HEEL_z', 'RIGHT_HEEL_visibility',
    'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_z', 'RIGHT_FOOT_INDEX_visibility'
]

# Feature extraction function from Mediapipe keypoints
def extract_features_from_keypoints(results):
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
    right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    left_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    right_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

    features = np.array([
        left_ankle.x, left_ankle.y, left_ankle.z, left_ankle.visibility,
        right_ankle.x, right_ankle.y, right_ankle.z, right_ankle.visibility,
        left_heel.x, left_heel.y, left_heel.z, left_heel.visibility,
        right_heel.x, right_heel.y, right_heel.z, right_heel.visibility,
        left_foot_index.x, left_foot_index.y, left_foot_index.z, left_foot_index.visibility,
        right_foot_index.x, right_foot_index.y, right_foot_index.z, right_foot_index.visibility
    ])
    return features

#store features, frame names, and predictions
features_list = []
frame_names = []
predictions = []

# Process frames from video C0834_Walker1
for frame_file in sorted(os.listdir(frame_directory)):
    if frame_file.endswith('.jpg'):
        frame_path = os.path.join(frame_directory, frame_file)
        frame = cv2.imread(frame_path)
        
        # Convert the frame to RGB (required by Mediapipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get the results
        results = pose.process(rgb_frame)
        
        # Extract features from the foot keypoints if landmarks are detected
        if results.pose_landmarks:
            feature_vector = extract_features_from_keypoints(results)
            features_list.append(feature_vector)
            frame_names.append(frame_file)
            
            # Make the prediction using the trained model
            dmatrix = xgb.DMatrix(np.array([feature_vector]), feature_names=expected_columns)
            prediction = model.predict(dmatrix)
            predictions.append(prediction[0])

# Adjust the threshold for foot contact prediction
threshold = 0.5  

# Output predictions for each frame and check for "No Foot Contact"
no_contact_count = 0
red_card_ranges = []
current_range_start = None

# Prepare output data for JSON
output_data = {
    'predictions': [],
    'red_card_ranges': []
}

# Track the sequence of frames with "No Foot Contact"
for idx, prediction in enumerate(predictions):
    frame = frame_names[idx]  # Get the frame name
    predicted_contact = 'Foot Contact' if prediction >= threshold else 'No Foot Contact'
    output_data['predictions'].append({
        'frame': frame,
        'prediction': predicted_contact
    })

    # If no foot contact, increment the count
    if prediction < threshold:
        no_contact_count += 1
        if current_range_start is None:  # Start a new range
            current_range_start = frame
    else:
        if no_contact_count >= 5:  # If there are 5 or more consecutive "No Foot Contact"
            red_card_ranges.append((current_range_start, frame_names[idx - 1]))  # Save the range of frames
        no_contact_count = 0
        current_range_start = None  # Reset range if foot contact is detected

# If the sequence ends with a red card (handle case where no contact ends at last frame)
if no_contact_count >= 5:
    red_card_ranges.append((current_range_start, frame_names[-1]))

# Add red card ranges to the output data
for start_frame, end_frame in red_card_ranges:
    output_data['red_card_ranges'].append({
        'start_frame': start_frame.split('_')[-1],  # Extract frame number
        'end_frame': end_frame.split('_')[-1]  # Extract frame number
    })

# Save the output to a local JSON file
output_json_path = '/tmp/C0834_output_with_red_card.json'  # Temporary path
with open(output_json_path, 'w') as json_file:
    json.dump(output_data, json_file, indent=4)

# Upload the JSON file to S3
output_s3_path = 'output/C0834_output_with_red_card.json'  # Specify your S3 path
s3_client.upload_file(output_json_path, bucket_name, output_s3_path)

print(f"Results uploaded to S3: s3://{bucket_name}/{output_s3_path}")
