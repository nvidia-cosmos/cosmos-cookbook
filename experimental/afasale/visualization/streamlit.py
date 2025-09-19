import argparse
import os
import sys

import streamlit as st

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Streamlit app for viewing videos with captions")
    parser.add_argument("--video_dir", required=True, help="Directory containing MP4 video files")
    parser.add_argument("--prompt_dir", required=True, help="Directory containing text prompt files")
    return parser.parse_args()

# Get arguments
args = parse_args()

# Configure page
st.set_page_config(page_title="Video Prompt Viewer", layout="wide")
st.title("🎬 MP4 Videos with Captions")

# Set directories from command line arguments
video_dir = args.video_dir
prompt_dir = args.prompt_dir

# Display the directories being used
st.info(f"Video directory: {video_dir}")
st.info(f"Prompt directory: {prompt_dir}")

# Get all videos
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

if not video_files:
    st.warning("No videos found in 'videos/' folder.")
else:
    cols = st.columns(5)  # 3 videos per row
    cnt = 150
    for idx, video_file in enumerate(video_files):
        cnt = cnt - 1
        if cnt == 0:
            break
        video_path = os.path.join(video_dir, video_file)

        # Match corresponding prompt file
        prompt_filename = os.path.splitext(video_file)[0] + ".txt"
        prompt_path = os.path.join(prompt_dir, prompt_filename)

        # Load prompt
        prompt = ""
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = "*No prompt found.*"

        # Display video and prompt
        with cols[idx % 4]:
            st.video(video_path)
            st.text(f"📝 {prompt}")
