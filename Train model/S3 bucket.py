import boto3

#Initialize S3 client
s3_client = boto3.client('s3')

# Define bucket name and object key
bucket_name = 
video_s3_key = f"frames/{video_name}.mp4"

# Check if the video exists in our S3 bucket
try:
    s3_client.head_object(Bucket=bucket_name, Key=video_s3_key)
    print(f"The video exists at: s3://{bucket_name}/{video_s3_key}")
except s3_client.exceptions.ClientError as e:
    print(f"Error: The video does not exist at s3://{bucket_name}/{video_s3_key}")
