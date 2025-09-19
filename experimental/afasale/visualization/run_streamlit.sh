#!/bin/bash

# Script to run the Streamlit video viewer app
# Usage: ./run_streamlit.sh <video_directory> <prompt_directory> [port]
# Example: ./run_streamlit.sh /path/to/videos /path/to/prompts 8501

# Check if at least two arguments are provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <video_directory> <prompt_directory> [port]"
    echo ""
    echo "Examples:"
    echo "  $0 /home/user/videos /home/user/captions        # Uses default port 8501"
    echo "  $0 /home/user/videos /home/user/captions 8080   # Uses custom port 8080"
    echo ""
    echo "The video_directory should contain .mp4 files"
    echo "The prompt_directory should contain corresponding .txt files"
    echo "The port is optional and defaults to 8501 (valid range: 1024-65535)"
    exit 1
fi

VIDEO_DIR="$1"
PROMPT_DIR="$2"
PORT="${3:-8501}"  # Use provided port or default to 8501

# Validate port number
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then
    echo "Error: Port must be a valid number between 1024 and 65535!"
    echo "Provided port: $PORT"
    exit 1
fi

# Check if directories exist
if [ ! -d "$VIDEO_DIR" ]; then
    echo "Error: Video directory '$VIDEO_DIR' does not exist!"
    exit 1
fi

if [ ! -d "$PROMPT_DIR" ]; then
    echo "Error: Prompt directory '$PROMPT_DIR' does not exist!"
    exit 1
fi

echo "Starting Streamlit app..."
echo "Video directory: $VIDEO_DIR"
echo "Prompt directory: $PROMPT_DIR"
echo "Port: $PORT"
echo ""

# Run the Streamlit app with the provided directories and port
streamlit run streamlit_mcq.py --server.port "$PORT" -- --videos_dir "$VIDEO_DIR" --captions_dir "$PROMPT_DIR"
