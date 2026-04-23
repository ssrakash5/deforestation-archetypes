import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import sys

def download_dir(prefix, local, bucket, client):
    """
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    - bucket: s3 bucket with target contents
    - client: initialized s3 client
    """
    paginator = client.get_paginator('list_objects_v2')
    print(f"Listing objects in s3://{bucket}/{prefix}...")
    
    count = 0
    for result in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                k = key['Key']
                # Construct local path, removing the prefix part to keep it clean if desired, 
                # or just mirroring the full structure. user asked to download "into my data folder"
                # so mirroring structure inside data folder is safest.
                # However, k includes 'spacenet/SN7_buildings/...', so joining with data will result in data/spacenet/SN7_buildings/...
                dest_pathname = os.path.join(local, k)
                
                if not os.path.exists(os.path.dirname(dest_pathname)):
                    os.makedirs(os.path.dirname(dest_pathname))
                
                if not k.endswith('/'):
                    print(f"Downloading {k}...")
                    try:
                        client.download_file(bucket, k, dest_pathname)
                        count += 1
                    except Exception as e:
                        print(f"Failed to download {k}: {e}")

    return count

if __name__ == '__main__':
    BUCKET_NAME = 'spacenet-dataset'
    PREFIX = 'spacenet/SN7_buildings/'
    LOCAL_DIR = '/content/drive/MyDrive/deforestation_archetypes/data'
    
    # Ensure boto3 is installed
    try:
        import boto3
    except ImportError:
        print("boto3 is not installed. Please run 'pip install boto3'")
        sys.exit(1)

    print(f"Starting download from s3://{BUCKET_NAME}/{PREFIX} to {LOCAL_DIR}...")
    
    # helper for checking directory existence
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    downloaded_count = download_dir(PREFIX, LOCAL_DIR, BUCKET_NAME, s3)
    print(f"Download completed. {downloaded_count} files downloaded.")
