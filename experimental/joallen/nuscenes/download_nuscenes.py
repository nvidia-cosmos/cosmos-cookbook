#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests",
#   "tqdm",
# ]
# [tool.uv]
# exclude-newer = "2025-07-31T00:00:00Z"
# ///

import argparse
import gzip
import hashlib
import json
import os
import tarfile

import requests
from tqdm import tqdm

download_files = {
    "v1.0-test_meta.tgz": "b0263f5c41b780a5a10ede2da99539eb",
    "v1.0-test_blobs.tgz": "e065445b6019ecc15c70ad9d99c47b33",
    "v1.0-trainval01_blobs.tgz": "cbf32d2ea6996fc599b32f724e7ce8f2",
    "v1.0-trainval02_blobs.tgz": "aeecea4878ec3831d316b382bb2f72da",
    "v1.0-trainval03_blobs.tgz": "595c29528351060f94c935e3aaf7b995",
    "v1.0-trainval04_blobs.tgz": "b55eae9b4aa786b478858a3fc92fb72d",
    "v1.0-trainval05_blobs.tgz": "1c815ed607a11be7446dcd4ba0e71ed0",
    "v1.0-trainval06_blobs.tgz": "7273eeea36e712be290472859063a678",
    "v1.0-trainval07_blobs.tgz": "46674d2b2b852b7a857d2c9a87fc755f",
    "v1.0-trainval08_blobs.tgz": "37524bd4edee2ab99678909334313adf",
    "v1.0-trainval09_blobs.tgz": "a7fcd6d9c0934e4052005aa0b84615c0",
    "v1.0-trainval10_blobs.tgz": "31e795f2c13f62533c727119b822d739",
    "v1.0-trainval_meta.tgz": "537d3954ec34e5bcb89a35d4f6fb0d4a",
}


def login(username: str, password: str):
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }

    # Use json.dumps() for correct JSON formatting
    data = json.dumps(
        {
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
            "AuthParameters": {"USERNAME": username, "PASSWORD": password},
            "ClientMetadata": {},
        }
    )

    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=data,
    )

    if response.status_code == 200:
        try:
            token = json.loads(response.content)["AuthenticationResult"]["IdToken"]
            return token
        except KeyError:
            print(
                "Authentication failed. 'AuthenticationResult' not found in the response."
            )
    else:
        print("Failed to login. Status code:", response.status_code)

    return None


def download_file(url: str, save_file: str, md5: str):
    response = requests.get(url, stream=True)
    if save_file.endswith(".tgz"):
        content_type = response.headers.get("Content-Type", "")
        if content_type == "application/x-tar":
            save_file = save_file.replace(".tgz", ".tar")
        elif content_type != "application/octet-stream":
            print("unknow content type", content_type)
            return save_file

    if os.path.exists(save_file):
        print(save_file, "has downloaded")
        # check md5
        md5obj = hashlib.md5()
        with open(save_file, "rb") as file:
            for chunk in file:
                md5obj.update(chunk)
        hash = md5obj.hexdigest()
        if hash != md5:
            print(save_file, "check md5 failed,download again")
        else:
            print(save_file, "check md5 success")
            return save_file

    file_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=save_file,
        ascii=True,
    )

    # save file & check md5
    md5obj = hashlib.md5()
    with open(save_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                md5obj.update(chunk)
                file.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()

    hash = md5obj.hexdigest()
    if hash != md5:
        print(save_file, "check md5 failed")
    else:
        print(save_file, "check md5 success")

    return save_file


def extract_tgz_to_original_folder(tgz_file_path):
    original_folder = os.path.dirname(tgz_file_path)
    print(f"Extracting {tgz_file_path} to {original_folder}")

    with gzip.open(tgz_file_path, "rb") as f_in:
        with tarfile.open(fileobj=f_in, mode="r") as tar:
            tar.extractall(original_folder)


def extract_tar_to_original_folder(tar_file_path):
    original_folder = os.path.dirname(tar_file_path)
    print(f"Extracting {tar_file_path} to {original_folder}")

    with tarfile.open(tar_file_path, "r") as tar:
        tar.extractall(original_folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--region", type=str, default="us", choices=["us", "asia"])
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    print("Logging in...")
    bearer_token = login(args.username, args.password)
    # set request header
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }

    print("Getting download urls...")
    download_data = {}
    for filename, md5 in download_files.items():
        api_url = f"https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/{filename}?region={args.region}&project=nuScenes"

        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            print(filename, "request success")
            download_url = response.json()["url"]
            download_data[filename] = [
                download_url,
                os.path.join(args.output_dir, filename),
                md5,
            ]
        else:
            print(f"request failed : {response.status_code}")
            print(response.text)

    if not args.no_download:
        print("Downloading files...")
        os.makedirs(args.output_dir, exist_ok=True)
        for output_name, (download_url, save_file, md5) in download_data.items():
            save_file = download_file(download_url, save_file, md5)
            download_data[output_name] = [download_url, save_file, md5]

    if not args.no_extract:
        print("Extracting files...")
        for output_name, (download_url, save_file, md5) in download_data.items():
            if output_name.endswith(".tgz"):
                extract_tgz_to_original_folder(save_file)
            elif output_name.endswith(".tar"):
                extract_tar_to_original_folder(save_file)
            else:
                print("unknow file type", output_name)

    print("Done!")


if __name__ == "__main__":
    main()
