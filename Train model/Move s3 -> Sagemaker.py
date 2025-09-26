import boto3
import os

# Initialize S3 client
s3_client = boto3.client('s3')

# S3 Bucket and folder details
bucket_name = 
video_name = 'C0834_Walker1' 
s3_folder = f'frames/{video_name}/'  
local_folder = f'/tmp/{video_name}/'  


if not os.path.exists(local_folder):
    os.makedirs(local_folder)

# Download frames from S3 to local folder
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
for obj in response.get('Contents', []):
    file_name = obj['Key'].split('/')[-1]
    local_file_path = os.path.join(local_folder, file_name)
    s3_client.download_file(bucket_name, obj['Key'], local_file_path)
    print(f"Downloaded {file_name} to {local_file_path}")

print(f"Frames for video {video_name} have been successfully downloaded to {local_folder}")
