#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a mini SutureBot dataset by copying a subset of episodes to a new folder,
then run convert_suturebot_to_lerobot_v3.py on it. Uses the same directory
structure expected by the conversion script (tissue_*/subtask/episode).

Run from the same directory as convert_suturebot_to_lerobot_v3.py (scripts/).
Does not modify convert_suturebot_to_lerobot_v3.py.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def discover_episodes(data_path: Path, tissue_filter: str | None):
    """Yield (tissue_dir, subtask_name, episode_name, episode_src_path)."""
    tissue_dirs = sorted(d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("tissue_"))
    if tissue_filter:
        tissue_dirs = [d for d in tissue_dirs if d.name == tissue_filter]
    for tissue_dir in tissue_dirs:
        for subtask_name in sorted(os.listdir(tissue_dir)):
            subtask_dir = tissue_dir / subtask_name
            if not subtask_dir.is_dir():
                continue
            for episode_name in sorted(os.listdir(subtask_dir)):
                episode_src = subtask_dir / episode_name
                if not episode_src.is_dir():
                    continue
                yield tissue_dir.name, subtask_name, episode_name, episode_src


def create_mini_dataset(
    source: Path,
    output: Path,
    max_episodes_per_subtask: int = 3,
    tissue: str = "tissue_1",
) -> int:
    """Copy a subset of episodes to output. Returns number of episodes copied."""
    source = source.resolve()
    output = output.resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    # Collect episodes to copy: (tissue, subtask, episode_name, src_path), limit per subtask
    subtask_counts: dict[tuple[str, str], int] = {}
    to_copy: list[tuple[str, str, str, Path]] = []
    for tissue_name, subtask_name, episode_name, episode_src in discover_episodes(source, tissue):
        key = (tissue_name, subtask_name)
        n = subtask_counts.get(key, 0)
        if n >= max_episodes_per_subtask:
            continue
        subtask_counts[key] = n + 1
        to_copy.append((tissue_name, subtask_name, episode_name, episode_src))

    if not to_copy:
        raise SystemExit(
            f"No episodes found under {source} for tissue={tissue}. "
            "Check that source is the dataset root containing tissue_*/ subdirs."
        )

    # Copy
    for tissue_name, subtask_name, episode_name, episode_src in to_copy:
        dest_dir = output / tissue_name / subtask_name / episode_name
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(episode_src, dest_dir)
        print(f"  {tissue_name}/{subtask_name}/{episode_name}")

    return len(to_copy)


def main():
    parser = argparse.ArgumentParser(description="Create mini SutureBot dataset and optionally run LeRobot conversion.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/ephemeral/data"),
        help="Dataset root containing tissue_* directories (default: /ephemeral/data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/ephemeral/data/SutureBot_mini"),
        help="Output directory for the mini dataset (default: /ephemeral/data/SutureBot_mini)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=3,
        help="Max episodes per subtask to copy (default: 3)",
    )
    parser.add_argument(
        "--tissue",
        default="tissue_1",
        help="Tissue directory to use, e.g. tissue_1 (default: tissue_1)",
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Only create the mini folder; do not run convert_suturebot_to_lerobot_v3.py",
    )
    parser.add_argument(
        "--repo-id",
        default="suturebot_lerobot_mini",
        help="Repo ID for LeRobot output (default: suturebot_lerobot_mini)",
    )
    args = parser.parse_args()

    print(f"Creating mini dataset: {args.source} -> {args.output}")
    print(f"  tissue={args.tissue}, max_episodes_per_subtask={args.max_episodes}")
    n = create_mini_dataset(
        args.source,
        args.output,
        max_episodes_per_subtask=args.max_episodes,
        tissue=args.tissue,
    )
    print(f"Copied {n} episodes.")

    if args.no_convert:
        print("Skipping conversion (--no-convert). Run convert_suturebot_to_lerobot_v3.py manually.")
        return 0

    script_dir = Path(__file__).resolve().parent
    convert_script = script_dir / "convert_suturebot_to_lerobot_v3.py"
    if not convert_script.exists():
        raise SystemExit(f"Conversion script not found: {convert_script}")

    print(f"Running: {convert_script.name} --data-path {args.output} --repo-id {args.repo_id}")
    rc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(convert_script),
            "--data-path",
            str(args.output),
            "--repo-id",
            args.repo_id,
        ],
        cwd=str(script_dir),
    ).returncode
    if rc != 0:
        raise SystemExit(rc)
    print("Mini LeRobot dataset written under HF_LEROBOT_HOME.")
    return 0


if __name__ == "__main__":
    main()
