import streamlit as st
import os
import json
from pathlib import Path
import glob

# Default configuration file
CONFIG_FILE = "experimental/afasale/visualization/ground_truth_annotation/config.json"
MCQ_QUESTIONS_FILE = "experimental/afasale/visualization/ground_truth_annotation/mcq_questions.json"

def load_config():
    """Load configuration from JSON file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error(f"Error reading {CONFIG_FILE}. Please check the JSON format.")
            return None
    else:
        # Create default config if it doesn't exist
        default_config = {
            "default_video_directory": "/path/to/your/videos",
            "responses_filename": "video_responses.json",
            "mcq_questions_file": "mcq_questions.json",
            "app_settings": {
                "videos_per_page_options": [5, 10, 20, 50],
                "default_videos_per_page": 10
            }
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=2)
        st.warning(f"Created default {CONFIG_FILE}. Please update the video directory path.")
        return default_config

def load_mcq_questions(mcq_file):
    """Load MCQ questions from JSON file"""
    if os.path.exists(mcq_file):
        try:
            with open(mcq_file, 'r') as f:
                data = json.load(f)
                return data.get("questions", [])
        except json.JSONDecodeError:
            st.error(f"Error reading {mcq_file}. Please check the JSON format.")
            return []
    else:
        st.error(f"MCQ questions file '{mcq_file}' not found.")
        return []

def get_video_files(video_dir):
    """Get list of all video files from specified directory"""
    if not os.path.exists(video_dir):
        return []
    
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.webm"]
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        video_files.extend(glob.glob(os.path.join(video_dir, ext.upper())))
    
    return sorted([os.path.basename(f) for f in video_files])

def initialize_responses(video_files, mcq_questions):
    """Initialize responses for all videos"""
    if 'all_responses' not in st.session_state:
        st.session_state.all_responses = {}
        for video in video_files:
            st.session_state.all_responses[video] = {}
            for mcq in mcq_questions:
                st.session_state.all_responses[video][mcq["key"]] = None

def export_responses_to_json(responses, video_files, mcq_questions):
    """Export responses to JSON format"""
    json_responses = []
    
    for video in video_files:
        video_responses = responses.get(video, {})
        
        # Only include videos that have at least one answer
        if any(answer is not None for answer in video_responses.values()):
            captions = []
            
            for mcq in mcq_questions:
                answer_key = mcq["key"]
                answer_value = video_responses.get(answer_key)
                
                if answer_value is not None:
                    # Convert answer back to letter format
                    answer_letter = chr(65 + mcq["options"].index(answer_value))
                    
                    question_text = f"{mcq['question']}"
                    for i, option in enumerate(mcq["options"]):
                        question_text += f" ({chr(65 + i)}) {option}"
                    
                    captions.append({
                        "question": question_text,
                        "answer": answer_letter
                    })
                else:
                    # Add empty placeholder for incomplete answers
                    question_text = f"{mcq['question']}"
                    for i, option in enumerate(mcq["options"]):
                        question_text += f" ({chr(65 + i)}) {option}"
                    
                    captions.append({
                        "question": question_text,
                        "answer": ""
                    })
            
            json_responses.append({
                "video": video,
                "captions": captions
            })
    
    return json_responses

def main():
    st.set_page_config(
        page_title="Video Annotation Tool - Configurable",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Annotation Tool - Configurable")
    st.markdown("**Configure video path and MCQ questions via JSON files**")
    
    # Load configuration
    config = load_config()
    if config is None:
        st.stop()
    
    # Configuration section
    with st.expander("⚙️ Configuration", expanded=False):
        st.subheader("Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Video Directory:**")
            current_video_dir = st.text_input(
                "Video Directory Path",
                value=config.get("default_video_directory", ""),
                help="Enter the full path to your video directory"
            )
            
            st.write("**MCQ Questions File:**")
            mcq_file = st.text_input(
                "MCQ Questions JSON File",
                value=config.get("mcq_questions_file", "mcq_questions.json"),
                help="Path to JSON file containing MCQ questions"
            )
        
        with col2:
            st.write("**Output Settings:**")
            responses_file = st.text_input(
                "Responses Output File",
                value=config.get("responses_filename", "video_responses.json"),
                help="Name of the output JSON file"
            )
            
            st.write("**Configuration Files:**")
            st.code(f"Config: {CONFIG_FILE}")
            st.code(f"MCQ: {mcq_file}")
        
        # Update config button
        if st.button("💾 Update Configuration"):
            config["default_video_directory"] = current_video_dir
            config["mcq_questions_file"] = mcq_file
            config["responses_filename"] = responses_file
            
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success("Configuration updated! Please refresh the page.")
            st.rerun()
    
    st.markdown("---")
    
    # Load MCQ questions
    mcq_questions = load_mcq_questions(config.get("mcq_questions_file", MCQ_QUESTIONS_FILE))
    if not mcq_questions:
        st.error("No MCQ questions loaded. Please check your MCQ questions file.")
        st.stop()
    
    # Display loaded questions info
    st.info(f"📋 Loaded {len(mcq_questions)} questions from {config.get('mcq_questions_file', MCQ_QUESTIONS_FILE)}")
    
    # Get video files
    video_dir = config.get("default_video_directory", "")
    video_files = get_video_files(video_dir)
    
    if not video_files:
        st.error(f"No video files found in: {video_dir}")
        st.write("**Supported formats:** MP4, AVI, MOV, MKV, WMV, FLV, WEBM")
        return
    
    # Initialize responses
    initialize_responses(video_files, mcq_questions)
    
    # Progress tracking in sidebar
    st.sidebar.title("📊 Progress Tracker")
    
    completed_videos = sum(1 for video in video_files 
                          if all(st.session_state.all_responses[video][mcq["key"]] is not None 
                                for mcq in mcq_questions))
    
    st.sidebar.metric("Completed Videos", f"{completed_videos}/{len(video_files)}")
    st.sidebar.progress(completed_videos / len(video_files) if len(video_files) > 0 else 0)
    
    # Video directory info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Current Settings")
    st.sidebar.write(f"**Video Dir:** {os.path.basename(video_dir)}")
    st.sidebar.write(f"**Total Videos:** {len(video_files)}")
    st.sidebar.write(f"**Questions:** {len(mcq_questions)}")
    
    # Export section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Responses")
    
    responses_filename = config.get("responses_filename", "video_responses.json")
    
    if st.sidebar.button("🔄 Generate JSON Preview", help="Preview the JSON structure"):
        preview_data = export_responses_to_json(st.session_state.all_responses, video_files, mcq_questions)
        st.sidebar.json(preview_data[:2])  # Show first 2 entries as preview
        st.sidebar.write(f"Total entries: {len(preview_data)}")
    
    if st.sidebar.button("💾 Export All Responses", type="primary"):
        final_responses = export_responses_to_json(st.session_state.all_responses, video_files, mcq_questions)
        
        # Save to file
        with open(responses_filename, 'w') as f:
            json.dump(final_responses, f, indent=2)
        
        st.sidebar.success(f"✅ Exported {len(final_responses)} video responses to {responses_filename}!")
        
        # Download button
        json_string = json.dumps(final_responses, indent=2)
        st.sidebar.download_button(
            label="📥 Download JSON File",
            data=json_string,
            file_name=responses_filename,
            mime="application/json"
        )
    
    # Main content - Video list
    st.header(f"Video List ({len(video_files)} videos)")
    
    # Initialize page state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Process videos in batches to avoid overwhelming the interface
    videos_per_page_options = config.get("app_settings", {}).get("videos_per_page_options", [5, 10, 20, 50])
    default_per_page = config.get("app_settings", {}).get("default_videos_per_page", 10)
    
    videos_per_page = st.selectbox(
        "Videos per page:", 
        videos_per_page_options, 
        index=videos_per_page_options.index(default_per_page) if default_per_page in videos_per_page_options else 1
    )
    
    total_pages = (len(video_files) + videos_per_page - 1) // videos_per_page
    
    # Ensure current page is within valid range
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
    
    # Page selection with session state
    current_page = st.selectbox(
        f"Page (1-{total_pages}):", 
        range(1, total_pages + 1), 
        index=st.session_state.current_page - 1,
        key="page_selector"
    )
    
    # Update session state when selectbox changes
    st.session_state.current_page = current_page
    
    start_idx = (current_page - 1) * videos_per_page
    end_idx = min(start_idx + videos_per_page, len(video_files))
    current_page_videos = video_files[start_idx:end_idx]
    
    # Display videos for current page
    for video_idx, video_name in enumerate(current_page_videos):
        global_idx = start_idx + video_idx
        
        st.markdown("---")
        
        # Video header with completion status
        responses = st.session_state.all_responses[video_name]
        completed = all(responses[mcq["key"]] is not None for mcq in mcq_questions)
        status_icon = "✅" if completed else "❌"
        
        st.subheader(f"{status_icon} Video {global_idx + 1}: {video_name}")
        
        # Create columns for video and questions
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Video player
            video_path = os.path.join(video_dir, video_name)
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.error(f"Video file not found: {video_path}")
            
            # Video info
            st.info(f"**File:** {video_name}\n**Index:** {global_idx + 1}/{len(video_files)}")
        
        with col2:
            st.write(f"**Answer all {len(mcq_questions)} questions for this video:**")
            
            # Questions for this video
            for i, mcq in enumerate(mcq_questions):
                current_answer = responses[mcq["key"]]
                
                st.write(f"**Q{i+1}:** {mcq['question']}")
                
                answer = st.radio(
                    "Select answer:",
                    options=mcq["options"],
                    index=mcq["options"].index(current_answer) if current_answer else 0,
                    key=f"{video_name}_q{i}",
                    format_func=lambda x, opts=mcq["options"]: f"({chr(65 + opts.index(x))}) {x}",
                    horizontal=True
                )
                
                # Update response
                st.session_state.all_responses[video_name][mcq["key"]] = answer
                
                st.markdown("<br>", unsafe_allow_html=True)
    
    # Page navigation
    if total_pages > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_page > 1:
                if st.button("⬅️ Previous Page"):
                    st.session_state.current_page = current_page - 1
                    st.rerun()
        
        with col2:
            st.write(f"Page {current_page} of {total_pages}")
        
        with col3:
            if current_page < total_pages:
                if st.button("➡️ Next Page"):
                    st.session_state.current_page = current_page + 1
                    st.rerun()
    
    # Summary at the bottom
    st.markdown("---")
    st.subheader("📋 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Videos", len(video_files))
    with col2:
        st.metric("Completed", completed_videos)
    with col3:
        remaining = len(video_files) - completed_videos
        st.metric("Remaining", remaining)
    with col4:
        st.metric("Questions per Video", len(mcq_questions))
    
    if completed_videos == len(video_files):
        st.success("🎉 All videos completed! You can now export your responses.")
    else:
        st.warning(f"⚠️ {remaining} videos still need to be completed.")

if __name__ == "__main__":
    main()
