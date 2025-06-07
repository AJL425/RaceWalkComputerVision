import os
import boto3
import json
import cv2
import mediapipe as mp
import xgboost as xgb
from urllib.parse import unquote_plus

# Initialize S3 client
s3_client = boto3.client('s3')

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define expected feature columns
expected_columns = [
    'LEFT_ANKLE_x', 'LEFT_ANKLE_y', 'LEFT_ANKLE_z', 'LEFT_ANKLE_visibility',
    'LEFT_HEEL_x', 'LEFT_HEEL_y', 'LEFT_HEEL_z', 'LEFT_HEEL_visibility',
    'LEFT_FOOT_INDEX_x', 'LEFT_FOOT_INDEX_y', 'LEFT_FOOT_INDEX_z', 'LEFT_FOOT_INDEX_visibility',
    'RIGHT_ANKLE_x', 'RIGHT_ANKLE_y', 'RIGHT_ANKLE_z', 'RIGHT_ANKLE_visibility',
    'RIGHT_HEEL_x', 'RIGHT_HEEL_y', 'RIGHT_HEEL_z', 'RIGHT_HEEL_visibility',
    'RIGHT_FOOT_INDEX_x', 'RIGHT_FOOT_INDEX_y', 'RIGHT_FOOT_INDEX_z', 'RIGHT_FOOT_INDEX_visibility'
]

def lambda_handler(event, context):
    model_path = '/tmp/xgboost-model.json'
    model_bucket = 'racewalk-ui-testing'
    model_key = 'models/xgboost-model.json'

    try:
        s3_client.download_file(model_bucket, model_key, model_path)
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise

    model = xgb.Booster()
    model.load_model(model_path)

    try:
        bucket_name = event['bucket']
        frame_prefix = event['frame_prefix']

        parts = frame_prefix.split('/')
        if len(parts) < 4:
            raise ValueError("Invalid frame_prefix structure")

        user_id = parts[1]
        project_name = parts[2]
        result_prefix = f"projects/{user_id}/{project_name}/results/"
        result_key = f"{result_prefix}{project_name}_output_with_red_card.json"

        # Skip if result already exists
        try:
            s3_client.head_object(Bucket=bucket_name, Key=result_key)
            print(f"✅ Results already exist: s3://{bucket_name}/{result_key}. Skipping inference.")
            return {"status": "skipped"}
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                raise

        print(f"🔍 Running model on {frame_prefix} -> Output: {result_prefix}")
        test_on_model(bucket_name, frame_prefix, result_prefix, model, project_name)

    except Exception as e:
        print(f"❌ Error: {e}")

    return {"status": "done"}


def extract_features_from_keypoints(results):
    lmk = mp_pose.PoseLandmark
    pose_lm = results.pose_landmarks.landmark
    return [
        pose_lm[lmk.LEFT_ANKLE].x, pose_lm[lmk.LEFT_ANKLE].y, pose_lm[lmk.LEFT_ANKLE].z, pose_lm[lmk.LEFT_ANKLE].visibility,
        pose_lm[lmk.LEFT_HEEL].x, pose_lm[lmk.LEFT_HEEL].y, pose_lm[lmk.LEFT_HEEL].z, pose_lm[lmk.LEFT_HEEL].visibility,
        pose_lm[lmk.LEFT_FOOT_INDEX].x, pose_lm[lmk.LEFT_FOOT_INDEX].y, pose_lm[lmk.LEFT_FOOT_INDEX].z, pose_lm[lmk.LEFT_FOOT_INDEX].visibility,
        pose_lm[lmk.RIGHT_ANKLE].x, pose_lm[lmk.RIGHT_ANKLE].y, pose_lm[lmk.RIGHT_ANKLE].z, pose_lm[lmk.RIGHT_ANKLE].visibility,
        pose_lm[lmk.RIGHT_HEEL].x, pose_lm[lmk.RIGHT_HEEL].y, pose_lm[lmk.RIGHT_HEEL].z, pose_lm[lmk.RIGHT_HEEL].visibility,
        pose_lm[lmk.RIGHT_FOOT_INDEX].x, pose_lm[lmk.RIGHT_FOOT_INDEX].y, pose_lm[lmk.RIGHT_FOOT_INDEX].z, pose_lm[lmk.RIGHT_FOOT_INDEX].visibility
    ]

def test_on_model(bucket_name, frame_prefix, result_prefix, model, project_name):
    # Check if the result already exists in S3
    result_key = f"{result_prefix}{project_name}_output_with_red_card.json"
    try:
        s3_client.head_object(Bucket=bucket_name, Key=result_key)
        print(f"✅ Results already exist: s3://{bucket_name}/{result_key}. Skipping inference.")
        return
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("🔍 No existing result found. Proceeding with inference.")
        else:
            raise

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=frame_prefix)
    frame_folder = '/tmp/frames/'
    os.makedirs(frame_folder, exist_ok=True)

    features_list = []
    frame_names = []

    print(f"Found {len(response.get('Contents', []))} objects in {frame_prefix}")

    for obj in response.get('Contents', []):
        key = obj['Key']
        if not key.endswith('.jpg'):
            continue

        file_name = os.path.basename(key)
        local_path = os.path.join(frame_folder, file_name)

        s3_client.download_file(bucket_name, key, local_path)
        frame = cv2.imread(local_path)
        if frame is None:
            continue

        print(f"Processing frame: {file_name}")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            feature_vector = extract_features_from_keypoints(results)
            features_list.append(feature_vector)
            frame_names.append(file_name)
        else:
            print(f"⚠️ No landmarks detected in frame {file_name}")

    if not features_list:
        print(f"❌ No valid features extracted from frames.")
        return

    predictions = []
    for fv in features_list:
        dmatrix = xgb.DMatrix([fv], feature_names=expected_columns)
        prediction = model.predict(dmatrix)
        predictions.append(prediction[0])

    print(f"Predictions: {predictions}")

    no_contact_count = 0
    red_card_ranges = []
    current_range_start = None

    output_data = {
        'predictions': [],
        'red_card_ranges': []
    }

    for idx, pred in enumerate(predictions):
        frame = frame_names[idx]
        contact = 'Foot Contact' if pred >= 0.5 else 'No Foot Contact'
        output_data['predictions'].append({'frame': frame, 'prediction': contact})

        if pred < 0.5:
            no_contact_count += 1
            if current_range_start is None:
                current_range_start = frame
        else:
            if no_contact_count >= 5:
                red_card_ranges.append((current_range_start, frame_names[idx - 1]))
            no_contact_count = 0
            current_range_start = None

    if no_contact_count >= 5:
        red_card_ranges.append((current_range_start, frame_names[-1]))

    output_data['red_card_ranges'] = red_card_ranges
    print(f"Red card ranges: {red_card_ranges}")

    # Upload the results to S3
    print(f"Uploading results to s3://{bucket_name}/{result_key}")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=result_key,
        Body=json.dumps(output_data, indent=4)
    )
    print(f"✅ Results uploaded to s3://{bucket_name}/{result_key}")
