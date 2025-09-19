#!/usr/bin/env python3
"""
Streamlit app to view video analysis results with thinking process and answers.

Usage:
    streamlit run streamlit_mcq.py -- --captions_dir /path/to/captions --videos_dir /path/to/videos
    OR
    streamlit run streamlit_mcq.py --server.port 8501 -- --captions_dir /path/to/captions --videos_dir /path/to/videos
"""

import streamlit as st
import re
from pathlib import Path
import os
import argparse
import sys

def parse_result_file(file_path):
    """Parse the result text file to extract components."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract video path
    video_match = re.search(r'Video: (.+)', content)
    video_path = video_match.group(1) if video_match else None
    
    # Extract prompt
    prompt_match = re.search(r'Prompt: (.+)', content)
    prompt = prompt_match.group(1) if prompt_match else None
    
    # Extract thinking process
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    
    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    
    return {
        'video_path': video_path,
        'prompt': prompt,
        'thinking': thinking,
        'answer': answer,
        'raw_content': content
    }

def find_result_files(directory):
    """Find all .txt result files in the directory."""
    if not directory or not Path(directory).exists():
        return []
    
    txt_files = []
    for file_path in Path(directory).rglob('*.txt'):
        if file_path.is_file():
            txt_files.append(file_path)
    
    return sorted(txt_files)

def calculate_answer_distribution(result_files):
    """Calculate the percentage distribution of answers A, B, C, D."""
    answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    total_valid_answers = 0
    
    for result_file in result_files:
        try:
            result_data = parse_result_file(result_file)
            answer = result_data.get('answer', '').strip().upper()
            
            # Check if answer is one of A, B, C, D
            if answer in answer_counts:
                answer_counts[answer] += 1
                total_valid_answers += 1
                
        except Exception as e:
            continue  # Skip files that can't be parsed
    
    # Calculate percentages
    answer_percentages = {}
    if total_valid_answers > 0:
        for option in ['A', 'B', 'C', 'D']:
            percentage = (answer_counts[option] / total_valid_answers) * 100
            answer_percentages[option] = {
                'count': answer_counts[option],
                'percentage': percentage
            }
    else:
        for option in ['A', 'B', 'C', 'D']:
            answer_percentages[option] = {'count': 0, 'percentage': 0.0}
    
    return answer_percentages, total_valid_answers

def get_sample_question(result_files):
    """Get a sample question from the first available result file."""
    for result_file in result_files:
        try:
            result_data = parse_result_file(result_file)
            if result_data.get('prompt'):
                return result_data['prompt']
        except Exception:
            continue
    return None

def filter_result_files_by_answer(result_files, answer_filter):
    """Filter result files based on selected answers."""
    if not answer_filter:  # If no answers selected, return empty list
        return []
    
    filtered_files = []
    for result_file in result_files:
        try:
            result_data = parse_result_file(result_file)
            answer = result_data.get('answer', '').strip().upper()
            
            # Include file if its answer is in the filter list
            if answer in answer_filter:
                filtered_files.append(result_file)
                
        except Exception:
            continue  # Skip files that can't be parsed
    
    return filtered_files

def find_video_file(result_file_path, videos_dir):
    """Find the corresponding video file for a result file."""
    # Get the base name of the result file without extension
    txt_path = Path(result_file_path)
    video_name = txt_path.stem + '.mp4'
    
    # Look for the video file in the videos directory
    videos_path = Path(videos_dir)
    if videos_path.exists():
        # First try exact path match in videos directory
        video_path = videos_path / video_name
        if video_path.exists():
            return str(video_path)
        
        # Then try recursive search in videos directory
        for video_file in videos_path.rglob(video_name):
            if video_file.is_file():
                return str(video_file)
    
    return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Streamlit Video Analysis Viewer')
    parser.add_argument('--captions_dir', 
                      type=str, 
                      default="dataset/mcq_captions/vq2_1/",
                      help='Path to the directory containing caption/result files (.txt)')
    parser.add_argument('--videos_dir', 
                      type=str, 
                      default="dataset/mcq_captions/vq2_1/",
                      help='Path to the directory containing video files (.mp4)')
    
    # Parse only known args to avoid conflicts with streamlit's args
    args, _ = parser.parse_known_args()
    return args

def main():
    # Parse command line arguments
    args = parse_args()
    
    st.set_page_config(
        page_title="Video Analysis Viewer",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Analysis Results Viewer")
    st.markdown("View all video analysis results with reasoning and answers in list view")
    
    # Sidebar for directory selection and controls
    with st.sidebar:
        st.header("📁 Directory Selection")
        
        # Directory inputs
        captions_dir = st.text_input(
            "Captions Directory", 
            value=args.captions_dir,
            help="Enter the path to the directory containing caption/result files (.txt)"
        )
        
        videos_dir = st.text_input(
            "Videos Directory", 
            value=args.videos_dir,
            help="Enter the path to the directory containing video files (.mp4)"
        )
        
        # Display controls
        st.header("⚙️ Display Settings")
        show_thinking_expanded = st.checkbox("Show thinking trace expanded", value=False)
        show_raw_content = st.checkbox("Show raw file content", value=False)
        videos_per_row = st.selectbox("Videos per row", [1, 2, 3], index=1)
        
        # Answer filter
        st.header("🔍 Filters")
        answer_filter = st.multiselect(
            "Filter by Answer",
            options=['A', 'B', 'C', 'D'],
            default=['A', 'B', 'C', 'D'],
            help="Select which answers to display"
        )
        
        if captions_dir:
            all_result_files = find_result_files(captions_dir)
            
            if all_result_files:
                # Apply answer filter
                result_files = filter_result_files_by_answer(all_result_files, answer_filter)
                
                st.success(f"Found {len(all_result_files)} total files, showing {len(result_files)} after filtering")
                
                if not result_files and answer_filter:
                    st.warning(f"No files found with answers: {', '.join(answer_filter)}")
            else:
                st.warning("No .txt files found in the captions directory")
                result_files = []
        else:
            result_files = []
    
    # Main content area - List view of all results
    if result_files:
        # Create header with filter info
        if answer_filter and len(answer_filter) < 4:
            filter_text = f" (filtered by: {', '.join(answer_filter)})"
            st.header(f"📋 Analysis Results ({len(result_files)} files{filter_text})")
        else:
            st.header(f"📋 Analysis Results ({len(result_files)} files)")
        
        # Display sample question
        sample_question = get_sample_question(result_files)
        if sample_question:
            st.subheader("❓ Question ❓")
            st.info(sample_question)
            st.markdown("---")
        
        # Calculate and display answer distribution
        answer_dist, total_answers = calculate_answer_distribution(result_files)
        
        if total_answers > 0:
            if answer_filter and len(answer_filter) < 4:
                st.subheader("📊 Answer Distribution (filtered results)")
            else:
                st.subheader("📊 Answer Distribution")
            
            # Create columns for the distribution display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Option A", 
                    value=f"{answer_dist['A']['count']}", 
                    delta=f"{answer_dist['A']['percentage']:.1f}%"
                )
            
            with col2:
                st.metric(
                    label="Option B", 
                    value=f"{answer_dist['B']['count']}", 
                    delta=f"{answer_dist['B']['percentage']:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="Option C", 
                    value=f"{answer_dist['C']['count']}", 
                    delta=f"{answer_dist['C']['percentage']:.1f}%"
                )
            
            with col4:
                st.metric(
                    label="Option D", 
                    value=f"{answer_dist['D']['count']}", 
                    delta=f"{answer_dist['D']['percentage']:.1f}%"
                )
            
            st.markdown(f"**Total valid answers:** {total_answers} out of {len(result_files)} files")
            st.markdown("---")
        else:
            st.warning("No valid answers (A, B, C, D) found in the result files")
        
        # Process all files and display in list view
        for i, result_file in enumerate(result_files):
            try:
                # Parse the result file
                result_data = parse_result_file(result_file)
                
                # Create a container for each result
                with st.container():
                    st.markdown("---")
                    st.subheader(f"📄 {result_file.name}")
                    
                    # Create columns based on user preference
                    if videos_per_row == 1:
                        # Single column layout with video on top
                        # Video section
                        video_path = find_video_file(result_file, videos_dir)
                        
                        if video_path:
                            st.video(video_path)
                        elif result_data['video_path'] and Path(result_data['video_path']).exists():
                            st.video(result_data['video_path'])
                        else:
                            video_name = Path(result_file).stem + '.mp4'
                            st.error(f"❌ Video file not found: {video_name}")
                        
                        # Question and Answer
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            if result_data['prompt']:
                                st.write("**Question:**")
                                st.info(result_data['prompt'])
                        with col2:
                            if result_data['answer']:
                                st.write("**Answer:**")
                                st.success(f"**{result_data['answer']}**")
                        
                    else:
                        # Multi-column layout
                        cols = st.columns([1, 2])  # Video, Content
                        
                        with cols[0]:
                            # Video section
                            video_path = find_video_file(result_file, videos_dir)
                            
                            if video_path:
                                st.video(video_path)
                            elif result_data['video_path'] and Path(result_data['video_path']).exists():
                                st.video(result_data['video_path'])
                            else:
                                st.error(f"❌ Video not found")
                        
                        with cols[1]:
                            # Question and Answer
                            if result_data['prompt']:
                                st.write("**Question:**")
                                st.info(result_data['prompt'])
                            
                            if result_data['answer']:
                                st.write("**Answer:**")
                                st.success(f"**{result_data['answer']}**")
                    
                    # Thinking process section
                    if result_data['thinking']:
                        with st.expander("🧠 View Thinking Process", expanded=show_thinking_expanded):
                            st.markdown(result_data['thinking'])
                    else:
                        st.warning("No thinking process found")
                    
                    # Raw content section (if enabled)
                    if show_raw_content:
                        with st.expander("📄 Raw File Content"):
                            st.text(result_data['raw_content'])
                
            except Exception as e:
                st.error(f"Error processing {result_file.name}: {str(e)}")
                continue
    
    else:
        st.info("👈 Please enter captions and videos directories in the sidebar to begin")
        
        # Show instructions
        st.markdown("""
        ### How to use:
        1. **Enter the captions directory** in the sidebar (where your .txt result files are located)
        2. **Enter the videos directory** in the sidebar (where your .mp4 video files are located)
        3. **Adjust display settings** for your preferred viewing experience
        4. **Scroll through all results** to see:
           - Videos with their analysis
           - Questions and answers
           - AI reasoning process (expandable)
        
        ### Display Options:
        - **Thinking process expanded**: Show/hide reasoning by default
        - **Raw file content**: Toggle display of original file content
        - **Videos per row**: Adjust layout density
        
        ### Filters:
        - **Filter by Answer**: Show only results with specific answers (A, B, C, D)
        
        ### Expected file structure:
        ```
        captions_directory/
        ├── scene-0077_CAM_FRONT.txt  # Analysis result
        ├── scene-0078_CAM_FRONT.txt
        └── ...
        
        videos_directory/
        ├── scene-0077_CAM_FRONT.mp4  # Corresponding video
        ├── scene-0078_CAM_FRONT.mp4
        └── ...
        ```
        
        ### Command line usage:
        ```bash
        streamlit run streamlit_mcq.py -- --captions_dir /path/to/captions --videos_dir /path/to/videos
        ```
        """)

if __name__ == "__main__":
    main()