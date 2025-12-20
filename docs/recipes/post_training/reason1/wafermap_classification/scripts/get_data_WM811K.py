# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""WM811K Wafer Map Data Preprocessing Script"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize


def parse_args():
    parser = argparse.ArgumentParser(description="WM811K Wafer Map Preprocessing")
    parser.add_argument(
        "-d", "--data-path", default="../WM811K.pkl", help="Path to WM811K pickle file"
    )
    parser.add_argument(
        "-o", "--output-dir", default="./WM811K_data", help="Output directory"
    )
    parser.add_argument(
        "-s",
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size (H W)",
    )
    parser.add_argument(
        "-t", "--train-count", type=int, default=100, help="Training images per type"
    )
    parser.add_argument(
        "-e", "--test-count", type=int, default=20, help="Testing images per type"
    )
    parser.add_argument("-c", "--colormap", default="viridis", help="Colormap")
    parser.add_argument("--vmin", type=float, default=0, help="Colormap min")
    parser.add_argument("--vmax", type=float, default=2, help="Colormap max")
    return parser.parse_args()


def clean_failure_type(df):
    """Clean failure type column."""

    def extract(val):
        if isinstance(val, (list, np.ndarray)):
            return str(val[0]).strip() if len(val) > 0 else "Unknown"
        return str(val).strip() if val else "Unknown"

    df["failureType_clean"] = df["failureType"].apply(extract)
    return df


def clean_train_test_label(df):
    """Clean train/test label column."""
    col = next((c for c in df.columns if c.lower() == "traintestlabel"), None)
    if not col:
        raise ValueError("trainTestLabel column not found")

    def extract(val):
        if isinstance(val, (list, np.ndarray)):
            items = [str(v).strip().lower() for v in val]
            if "training" in items:
                return "Training"
            if "test" in items:
                return "Test"
        elif isinstance(val, str):
            v = val.strip().lower()
            if v == "training":
                return "Training"
            if v == "test":
                return "Test"
        return "Unknown"

    df["trainTestLabel_clean"] = df[col].apply(extract)
    return df[df["trainTestLabel_clean"] != "Unknown"]


def save_wafer_image(wafer_map, save_path, args):
    """Resize and save wafer map as image."""
    resized = resize(
        wafer_map,
        tuple(args.image_size),
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    plt.imsave(save_path, resized, cmap=args.colormap, vmin=args.vmin, vmax=args.vmax)


def process_defect_type(wafer_maps, defect_type, args):
    """Process and save wafer maps for a defect type."""
    total = len(wafer_maps)
    train_n = min(args.train_count, total)
    test_n = min(args.test_count, total - train_n)
    print(f"  {defect_type}: {train_n} train, {test_n} test (total: {total})")

    train_saved = test_saved = 0
    for i in range(train_n):
        try:
            path = os.path.join(
                args.output_dir, "train", defect_type, f"wafermap_{defect_type}_{i}.png"
            )
            save_wafer_image(wafer_maps[i], path, args)
            train_saved += 1
        except Exception as e:
            print(f"    Error train {i}: {e}")

    for i in range(test_n):
        try:
            idx = train_n + i
            path = os.path.join(
                args.output_dir,
                "test",
                defect_type,
                f"wafermap_{defect_type}_{idx}.png",
            )
            save_wafer_image(wafer_maps[idx], path, args)
            test_saved += 1
        except Exception as e:
            print(f"    Error test {idx}: {e}")

    return train_saved, test_saved


def main():
    args = parse_args()
    print(f"Loading: {args.data_path}")

    df = pd.read_pickle(args.data_path)
    df = clean_failure_type(df)
    df = clean_train_test_label(df)

    # Group by failure type and get all types
    groups = {ft: g for ft, g in df.groupby("failureType")}
    defect_types = list(groups.keys())
    print(f"Found {len(defect_types)} defect types: {defect_types}")

    # Create directories for all defect types
    for split in ["train", "test"]:
        for dt in defect_types:
            os.makedirs(os.path.join(args.output_dir, split, dt), exist_ok=True)

    # Process all defect types
    total_train = total_test = 0
    for dt in defect_types:
        wafer_maps = groups[dt]["waferMap"].tolist()
        if wafer_maps:
            t, e = process_defect_type(wafer_maps, dt, args)
            total_train += t
            total_test += e

    print(f"\nDone! Train: {total_train}, Test: {total_test}")


if __name__ == "__main__":
    main()
