#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Union
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import shutil


class DataConverter:
    def __init__(self, aws_profile: str = "lha-share"):
        """Initialize the converter with AWS profile."""
        self.aws_profile = aws_profile
        self.s3_client = None
        self._init_s3_client()
    
    def _init_s3_client(self):
        """Initialize S3 client with the specified AWS profile."""
        try:
            session = boto3.Session(profile_name=self.aws_profile)
            self.s3_client = session.client('s3')
            print(f"Initialized S3 client with profile: {self.aws_profile}")
        except Exception as e:
            print(f"Warning: Could not initialize S3 client with profile {self.aws_profile}: {e}")
            print("Will only work with local paths.")
    
    def _is_s3_path(self, path: str) -> bool:
        """Check if path is an S3 path."""
        return path.startswith('s3://')
    
    def _parse_s3_path(self, s3_path: str) -> tuple:
        """Parse S3 path into bucket and key."""
        path = s3_path.replace('s3://', '')
        parts = path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        return bucket, key
    
    def _list_files(self, path: str, extension: str = None) -> list:
        """List files in a directory (local or S3)."""
        if self._is_s3_path(path):
            return self._list_s3_files(path, extension)
        else:
            return self._list_local_files(path, extension)
    
    def _list_s3_files(self, s3_path: str, extension: str = None) -> list:
        """List files in S3 directory."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket, prefix = self._parse_s3_path(s3_path)
        if not prefix.endswith('/'):
            prefix += '/'
        
        files = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key != prefix:  # Skip the directory itself
                            if extension is None or key.endswith(extension):
                                files.append(f"s3://{bucket}/{key}")
        except ClientError as e:
            print(f"Error listing S3 files: {e}")
        
        return files
    
    def _list_local_files(self, path: str, extension: str = None) -> list:
        """List files in local directory."""
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"Warning: Path {path} does not exist")
            return []
        
        pattern = f"*{extension}" if extension else "*"
        return [str(f) for f in path_obj.glob(pattern) if f.is_file()]
    
    def _copy_file(self, src: str, dst: str):
        """Copy file from source to destination (supports local and S3)."""
        src_is_s3 = self._is_s3_path(src)
        dst_is_s3 = self._is_s3_path(dst)
        
        if src_is_s3 and dst_is_s3:
            self._copy_s3_to_s3(src, dst)
        elif src_is_s3 and not dst_is_s3:
            self._download_s3_file(src, dst)
        elif not src_is_s3 and dst_is_s3:
            self._upload_file_to_s3(src, dst)
        else:
            self._copy_local_file(src, dst)
    
    def _copy_s3_to_s3(self, src_s3: str, dst_s3: str):
        """Copy file from S3 to S3."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        src_bucket, src_key = self._parse_s3_path(src_s3)
        dst_bucket, dst_key = self._parse_s3_path(dst_s3)
        
        copy_source = {'Bucket': src_bucket, 'Key': src_key}
        self.s3_client.copy(copy_source, dst_bucket, dst_key)
    
    def _download_s3_file(self, s3_path: str, local_path: str):
        """Download file from S3 to local."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket, key = self._parse_s3_path(s3_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3_client.download_file(bucket, key, local_path)
    
    def _upload_file_to_s3(self, local_path: str, s3_path: str):
        """Upload local file to S3."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket, key = self._parse_s3_path(s3_path)
        self.s3_client.upload_file(local_path, bucket, key)
    
    def _copy_local_file(self, src: str, dst: str):
        """Copy local file to local."""
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    
    def _read_file(self, path: str) -> str:
        """Read file content (local or S3)."""
        if self._is_s3_path(path):
            return self._read_s3_file(path)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _read_s3_file(self, s3_path: str) -> str:
        """Read file content from S3."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket, key = self._parse_s3_path(s3_path)
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')
    
    def _write_file(self, path: str, content: str):
        """Write content to file (local or S3)."""
        if self._is_s3_path(path):
            self._write_s3_file(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _write_s3_file(self, s3_path: str, content: str):
        """Write content to S3 file."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        bucket, key = self._parse_s3_path(s3_path)
        self.s3_client.put_object(Bucket=bucket, Key=key, Body=content.encode('utf-8'))
    
    def _ensure_directory(self, path: str):
        """Ensure directory exists (for S3, this is a no-op)."""
        if not self._is_s3_path(path):
            os.makedirs(path, exist_ok=True)
    
    def copy_videos(self, input_dir: str, output_dir: str):
        """Copy .mp4 files from clips/ to videos/."""
        clips_dir = f"{input_dir.rstrip('/')}/clips"
        videos_dir = f"{output_dir.rstrip('/')}/videos"
        
        print(f"Copying videos from {clips_dir} to {videos_dir}")
        self._ensure_directory(videos_dir)
        
        mp4_files = self._list_files(clips_dir, '.mp4')
        
        for mp4_file in mp4_files:
            filename = os.path.basename(mp4_file)
            dst_path = f"{videos_dir}/{filename}"
            
            try:
                self._copy_file(mp4_file, dst_path)
                print(f"Copied: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")
    
    def extract_captions(self, input_dir: str, output_dir: str):
        """Extract captions from JSON files and save as text files."""
        metas_input_dir = f"{input_dir.rstrip('/')}/metas/v0"
        metas_output_dir = f"{output_dir.rstrip('/')}/metas"
        
        print(f"Extracting captions from {metas_input_dir} to {metas_output_dir}")
        self._ensure_directory(metas_output_dir)
        
        json_files = self._list_files(metas_input_dir, '.json')
        
        for json_file in json_files:
            try:
                # Read and parse JSON file
                content = self._read_file(json_file)
                data = json.loads(content)
                
                # Extract qwen_caption from first window (same logic as extract_captions.py)
                if 'windows' in data and len(data['windows']) > 0:
                    caption = data['windows'][0].get('qwen_caption', '')
                    
                    # Create output text file
                    filename = os.path.basename(json_file)
                    txt_filename = f"{os.path.splitext(filename)[0]}.txt"
                    dst_path = f"{metas_output_dir}/{txt_filename}"
                    
                    # Write caption to text file
                    self._write_file(dst_path, caption)
                    print(f"Extracted caption: {txt_filename}")
                else:
                    print(f"Warning: No windows found in {os.path.basename(json_file)}")
                    
            except Exception as e:
                print(f"Error processing {os.path.basename(json_file)}: {e}")
    
    def copy_embeddings(self, input_dir: str, output_dir: str):
        """Copy .pickle files from iv2_embd/ to t5_xxl/."""
        embd_input_dir = f"{input_dir.rstrip('/')}/iv2_embd"
        embd_output_dir = f"{output_dir.rstrip('/')}/t5_xxl"
        
        print(f"Copying embeddings from {embd_input_dir} to {embd_output_dir}")
        self._ensure_directory(embd_output_dir)
        
        pickle_files = self._list_files(embd_input_dir, '.pickle')
        
        for pickle_file in pickle_files:
            filename = os.path.basename(pickle_file)
            dst_path = f"{embd_output_dir}/{filename}"
            
            try:
                self._copy_file(pickle_file, dst_path)
                print(f"Copied embedding: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")
    
    def convert(self, input_dir: str, output_dir: str):
        """Convert data from zetta format to cosmos format."""
        print(f"Converting data from {input_dir} to {output_dir}")
        print("=" * 60)
        
        # Copy videos
        self.copy_videos(input_dir, output_dir)
        print()
        
        # Extract captions
        self.extract_captions(input_dir, output_dir)
        print()
        
        # Copy embeddings
        self.copy_embeddings(input_dir, output_dir)
        print()
        
        print("Conversion completed successfully!")


# Script to convert data from Cosmos Curator (zetta) format to cosmos oss format
# python scripts/convert_data_zetta_to_cosmos.py --input_dir s3://lha-share/data/zetta/metropolis/train/v0 --output_dir s3://lha-share/data/cosmos/metropolis/train/v0 --aws-profile lha-share

def main():
    parser = argparse.ArgumentParser(
        description="Convert data from zetta format to cosmos format"
    )
    parser.add_argument(
        "--input_dir",
        help="Input directory (local path or s3://bucket/path)"
    )
    parser.add_argument(
        "--output_dir", 
        help="Output directory (local path or s3://bucket/path)"
    )
    parser.add_argument(
        "--aws-profile",
        default="lha-share",
        help="AWS profile to use (default: lha-share)"
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = DataConverter(aws_profile=args.aws_profile)
    
    # Run conversion
    try:
        converter.convert(args.input_dir, args.output_dir)
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
