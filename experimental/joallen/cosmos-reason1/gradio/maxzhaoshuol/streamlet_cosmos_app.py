# http://maxzhaoshuol.nvidia.com:8501/

import os
import tempfile

import streamlit as st

from dev.vllm_inference_intern import MODEL_TO_IP_DICT as INTERN_MODEL_TO_IP_DICT
from dev.vllm_inference_intern import inference as intern_inference

# Import standalone inference modules
from dev.vllm_inference_qwen import MODEL_TO_IP_DICT as QWEN_MODEL_TO_IP_DICT
from dev.vllm_inference_qwen import inference as qwen_inference

# Combine model configurations with family prefixes for clarity
MODEL_TO_IP_DICT = {}
MODEL_FAMILY_MAPPING = {}

# Add Qwen models with family prefix
for model_name, ip_dict in QWEN_MODEL_TO_IP_DICT.items():
    prefixed_name = f"Qwen: {model_name}"
    MODEL_TO_IP_DICT[prefixed_name] = ip_dict
    MODEL_FAMILY_MAPPING[prefixed_name] = "qwen"

# Add Intern models with family prefix  
for model_name, ip_dict in INTERN_MODEL_TO_IP_DICT.items():
    prefixed_name = f"Intern: {model_name}"
    MODEL_TO_IP_DICT[prefixed_name] = ip_dict
    MODEL_FAMILY_MAPPING[prefixed_name] = "intern"


def inference_multi_model(media_path, user_prompt, system_prompt, media_type="video", max_num_vision_tokens=None, timestamp_video=False, fps_value=4, temperature=0.01, seed=None):
    """
    Run inference on all available models simultaneously and return results from each
    """
    import concurrent.futures
    
    results = {}
    
    def single_model_inference(model_name):
        """Run inference on a single model using appropriate inference function"""
        try:
            # Determine model family and use appropriate inference function
            model_family = MODEL_FAMILY_MAPPING.get(model_name, "qwen")
            
            # Get the original model name without prefix
            original_model_name = model_name.split(": ", 1)[1] if ": " in model_name else model_name
            
            if model_family == "intern":
                # Use Intern inference with its specific parameters (use defaults for multi-model)
                response = intern_inference(
                    media_path=media_path,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model_name=original_model_name,
                    media_type=media_type,
                    # Intern-specific parameters with defaults for multi-model comparison
                    temperature=temperature,
                    max_tokens=2048,
                    enable_thinking=True,
                    use_tools=False,
                    seed=seed
                )
            else:
                # Use Qwen inference with its specific parameters
                response = qwen_inference(
                    media_path=media_path,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model_name=original_model_name,
                    media_type=media_type,
                    max_num_vision_tokens=max_num_vision_tokens,
                    timestamp_video=timestamp_video,
                    fps_value=fps_value,
                    temperature=temperature,
                    seed=seed
                )
            
            return {"success": True, "response": response}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Use ThreadPoolExecutor to run inference on all models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_TO_IP_DICT)) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(single_model_inference, model_name): model_name 
            for model_name in MODEL_TO_IP_DICT.keys()
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results[model_name] = result
            except Exception as e:
                results[model_name] = {"success": False, "error": f"Future execution error: {str(e)}"}
    
    return results

def main():
    st.set_page_config(
        page_title="Multi-Model Video/Image Analysis",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Model Video/Image Analysis")
    st.markdown("Upload a video or image to get AI-powered analysis using Qwen or Intern models!")
    
    # Initialize session state variables
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = ""
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model/Port selection
        st.header("Model Selection")
        
        # Get model options from MODEL_TO_IP_DICT and add comparison option
        model_options = list(MODEL_TO_IP_DICT.keys()) + ["Compare All Models"]
        
        # Initialize session state for selected model
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_options[0]  # Default to first option
        
        selected_model = st.selectbox(
            "Choose model/port:",
            options=model_options,
            index=model_options.index(st.session_state.selected_model),
            help=f"Select which model endpoint to use or compare all models. Available endpoints: {MODEL_TO_IP_DICT}"
        )
        
        # Update session state
        st.session_state.selected_model = selected_model
        
        # Display selected info
        if selected_model == "Compare All Models":
            st.info("🔄 Will query all models simultaneously")
            # Show all models that will be queried
            st.write("**Models to query:**")
            for model_name, ip_port in MODEL_TO_IP_DICT.items():
                st.write(f"• {model_name} ({ip_port})")
        else:
            selected_port = MODEL_TO_IP_DICT[selected_model]
            st.info(f"Using port: {selected_port}")
        
        # System prompt configuration
        st.header("System Prompt")
        
        # Predefined system prompts
        system_prompt_options = {
            "Default": "You are a helpful video analyzer. Please answer the question.",
            "Robot": "You are observing a video showing embodied agents (robots or humans) demonstration. Please answer the question.",
            "AV": "You are the person driving the vehicle. Please answer the question from the first-person view.",
            "Prompt upsapmler": """You are a prompt optimization specialist whose goal is to rewrite the user's input prompts into high-quality English prompts by referring to the details of the user's input images, making them more complete and expressive while maintaining the original meaning. You need to integrate the content of the user's photo with the input prompt for the rewrite, strictly adhering to the formatting of the examples provided.
Task Requirements:
1. For overly brief user inputs, reasonably infer and supplement details without changing the original meaning, making the image more complete and visually appealing;
2. Improve the characteristics of the main subject in the user's description (such as appearance, actions, expression, quantity, ethnicity, posture, surrounding environment etc.), rendering style, spatial relationships, and camera angles;
3. The overall output should be in English, retaining original text in quotes and book titles as well as important input information without rewriting them;
4. The prompt should match the userâ€™s intent and provide a precise and detailed style description. If the user has not specified a style, you need to carefully analyze the style of the user's provided photo and use that as a reference for rewriting;
5. You need to emphasize movement information in the input and different camera angles;
6. Your output should convey natural movement attributes, incorporating natural actions related to the described subject category, using simple and direct verbs as much as possible;
7. You should reference the detailed information in the image, such as character actions, clothing, backgrounds, and emphasize the details in the photo;
8. Control the rewritten prompt to around 80-100 words.
9. No matter what language the user inputs, you must always output in English.
Example of the rewritten English prompt:
1. A Japanese fresh film-style video of a young East Asian girl with double braids sitting by the boat. The girl wears a white square collar puff sleeve dress, decorated with pleats and buttons. She has fair skin, delicate features, and slightly melancholic eyes, staring directly at the camera. Her hair falls naturally, with bangs covering part of her forehead. She rests her hands on the boat, appearing natural and relaxed. The background features a blurred outdoor scene, with hints of blue sky, mountains, and some dry plants. The video has a vintage film texture. A medium shot of a seated portrait.
2. An anime illustration in vibrant thick painting style of a white girl with cat ears holding a folder, showing a slightly dissatisfied expression. She has long dark purple hair and red eyes, wearing a dark gray skirt and a light gray top with a white waist tie and a name tag in bold Chinese characters that says ""Ziyang"". The background has a light yellow indoor tone, with faint outlines of some furniture visible. A pink halo hovers above her head, in a smooth Japanese cel-shading style. A close-up shot from a slightly elevated perspective.
3. CG game concept digital art featuring a huge crocodile with its mouth wide open, with trees and thorns growing on its back. The crocodile's skin is rough and grayish-white, resembling stone or wood texture. Its back is lush with trees, shrubs, and thorny protrusions. With its mouth agape, the crocodile reveals a pink tongue and sharp teeth. The background features a dusk sky with some distant trees, giving the overall scene a dark and cold atmosphere. A close-up from a low angle.
4. In the style of an American drama promotional poster, Walter White sits in a metal folding chair wearing a yellow protective suit, with the words ""Breaking Bad"" written in sans-serif English above him, surrounded by piles of dollar bills and blue plastic storage boxes. He wears glasses, staring forward, dressed in a yellow jumpsuit, with his hands resting on his knees, exuding a calm and confident demeanor. The background shows an abandoned, dim factory with light filtering through the windows. There's a noticeable grainy texture. A medium shot with a straight-on close-up of the character.
Directly output the rewritten English text.""",
            "Critic": """"You are a helpful video analyzer. The goal is to identify artifacts and anomalies in the video.
Analyze the video carefully and answer the question according to the following template:

<think>
<overview>
[Brief description of the video.]
</overview>

<component name=""Component 1 Name"">
<analysis>
[Analysis or reasoning about this component.]
</analysis>
<anomaly>Yes | No</anomaly>
</component>

<component name=""Component 2 Name"">
<analysis>
[Analysis or reasoning about this component.]
</analysis>
<anomaly>Yes | No</anomaly>
</component>

<!-- Add more components as needed -->
</think>

<answer>
[Whether the video contains anomalies or artifacts. Answer ""Yes"" or ""No"".]
</answer>""",
            "Temporal caption (json)": """You are a helpful video analyzer. Please provide captions of all the events in the video with timestamps using the following format:
[
  {
    "start": <start time>,
    "end": <end time>,
    "caption": <caption of event 1>,
  },
  {
    "start": <start time>,
    "end": <end time>,
    "caption": <caption of event 2>,
  },
]""",
            "Temporal caption (text)": """Please provide captions of all the events in the video with timestamps using the following format:
<start time> <end time> caption of event 1.\n<start time> <end time> caption of event 2.\n
At each frame, the timestamp is embedded at the bottom of the video. You need to extract the timestamp and answer the user question.
""",
            "Custom...": ""
        }
        
        selected_prompt_type = st.selectbox(
            "Choose a system prompt type:",
            options=list(system_prompt_options.keys()),
            index=0,
            help="Select a predefined system prompt or choose 'Custom...' to write your own"
        )
        
        # Reasoning toggle
        enable_reasoning = st.checkbox(
            "Enable Reasoning Format",
            value=False,
            help="Add structured reasoning format with <think> and <answer> tags to the response"
        )
        
        # Get base system prompt
        if selected_prompt_type == "Custom...":
            base_system_prompt = st.text_area(
                "Enter your custom system prompt:",
                value="",
                height=150,
                placeholder="Type your custom system prompt here...",
                help="Define how the AI should respond to your questions"
            )
        else:
            base_system_prompt = st.text_area(
                "Selected system prompt:",
                value=system_prompt_options[selected_prompt_type],
                height=150,
                help="You can edit this prompt if needed"
            )
        
        # Add reasoning format if enabled
        if enable_reasoning:
            reasoning_instruction = " Answer the question with provided options in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
            system_prompt = base_system_prompt + reasoning_instruction
            
            # Show the final prompt with reasoning added
            st.text_area(
                "Final system prompt (with reasoning):",
                value=system_prompt,
                height=100,
                disabled=True,
                help="This is the final prompt that will be sent to the model"
            )
        else:
            system_prompt = base_system_prompt
        


        # Vision processing configuration
        st.header("Vision Processing")
        
        # Determine if we're using Intern models for this selection
        is_intern_selected = False
        is_comparison_mode = selected_model == "Compare All Models"
        
        if not is_comparison_mode:
            model_family = MODEL_FAMILY_MAPPING.get(selected_model, "qwen")
            is_intern_selected = (model_family == "intern")
        
        # Show different controls based on model family
        if is_intern_selected:
            # Intern-specific controls
            st.info("🧠 **Intern-S1 Model Settings**")
            
            max_tokens = st.number_input(
                "Max Tokens:",
                min_value=100,
                max_value=4096,
                value=2048,
                step=128,
                help="Maximum number of tokens to generate in the response."
            )
            
            enable_thinking = st.checkbox(
                "Enable Thinking Mode",
                value=True,
                help="Enable the model's reasoning capabilities for more detailed analysis."
            )
            
            use_tools = st.checkbox(
                "Enable Tool Calling",
                value=False,
                help="Enable tool calling capabilities for scientific analysis tasks."
            )
            
            # Set defaults for variables not used by Intern
            max_num_vision_tokens = None
            timestamp_video = False
            fps_value = 4
            
        elif is_comparison_mode:
            # Show both sets of controls for comparison mode
            st.info("🔄 **Multi-Model Comparison Settings**")
            st.write("Both Qwen and Intern model parameters will be used as appropriate.")
            
            # Qwen parameters
            st.write("**Qwen Model Parameters:**")
            max_num_vision_tokens = st.number_input(
                "Max Vision Tokens (Qwen):",
                min_value=0,
                max_value=12_800,
                value=8192,
                step=128,
                help="Maximum number of vision tokens for Qwen models."
            )
            
            timestamp_video = st.checkbox(
                "Timestamp Video (Qwen)",
                value=False,
                help="Add timestamps to video processing for Qwen models."
            )
            
            fps_value = st.number_input(
                "FPS (Qwen):",
                min_value=1,
                max_value=60,
                value=4,
                step=1,
                help="Frames per second for Qwen video processing."
            )
            
            # Intern parameters (will be used with defaults)
            st.write("**Intern Model Parameters:**")
            st.write("- Max Video Frames: 8")
            st.write("- Max Tokens: 2048") 
            st.write("- Thinking Mode: Enabled")
            st.write("- Tool Calling: Disabled")
            
        else:
            # Qwen-specific controls
            st.info("🤖 **Qwen Model Settings**")
            
            max_num_vision_tokens = st.number_input(
                "Max Vision Tokens:",
                min_value=0,
                max_value=12_800,
                value=8192,
                step=128,
                help="Maximum number of vision tokens to process. Higher values allow more detailed analysis but take more resources."
            )
            
            timestamp_video = st.checkbox(
                "Timestamp Video",
                value=False,
                help="Add timestamps to video processing for temporal understanding."
            )
            
            fps_value = st.number_input(
                "FPS:",
                min_value=1,
                max_value=60,
                value=4,
                step=1,
                help="Frames per second for video processing. Higher values provide more temporal detail but use more resources."
            )
        
        temperature = st.number_input(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            format="%.2f",
            help="Controls randomness in responses. Lower values (closer to 0) make output more deterministic and focused. Higher values make output more creative and varied."
        )
        
        # Seed control
        use_seed = st.checkbox(
            "Use Seed for Reproducible Results",
            value=False,
            help="Enable to set a specific seed for reproducible generation results."
        )
        
        seed = None
        if use_seed:
            seed = st.number_input(
                "Seed:",
                min_value=0,
                max_value=2**32-1,
                value=42,
                step=1,
                help="Seed value for reproducible results. Same seed will produce same output."
            )
        
        # File upload
        st.header("Upload Media")
        uploaded_file = st.file_uploader(
            "Choose a video or image file",
            type=['mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Supported video formats: MP4, AVI, MOV, MKV\nSupported image formats: JPG, JPEG, PNG, BMP, WEBP"
        )
    
    # Main content area
    if uploaded_file is not None:
        # Check if a new file was uploaded and clear the question if so
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None
            
        # Detect new file upload
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_uploaded_file != current_file_id:
            st.session_state.last_uploaded_file = current_file_id
            st.session_state.selected_question = ""  # Clear the question
            st.rerun()
        
        # Determine media type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        is_video = file_extension in ['mp4', 'avi', 'mov', 'mkv']
        media_type = "video" if is_video else "image"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Display media in two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"📹 Uploaded {media_type.title()}")
            
            if is_video:
                # Display video with controls
                st.video(uploaded_file)
                st.info(f"Video file: {uploaded_file.name}")
            else:
                # Display image
                st.image(uploaded_file, caption=f"Uploaded image: {uploaded_file.name}")
        
        with col2:
            st.subheader("💬 Ask a Question")
            
            # Show example questions in an expander
            with st.expander("💡 Example Questions"):
                example_questions = [
                    "What are the potential safety hazards?",
                    "Describe what is happening in this media.",
                    "What objects do you see?",
                    "What actions are being performed?",
                    "Are there any people in this media?",
                    "What is the robot doing?",
                    "Is this demonstration successful?",
                    "What could be improved in this process?"
                ]
                
                for i, question in enumerate(example_questions, 1):
                    if st.button(f"{question}", key=f"example_{i}", use_container_width=True):
                        st.session_state.selected_question = question
                        st.rerun()
            
            # Main question input
            user_prompt = st.text_area(
                "Enter your question:",
                value=st.session_state.selected_question,
                placeholder="Type your question here... (e.g., What are the potential safety hazards? Describe what is happening in this media.)",
                height=100,
                key="question_input"
            )
            
            # Analysis button
            if st.button("🔍 Analyze Media", type="primary", use_container_width=True):
                if user_prompt:
                    if selected_model == "Compare All Models":
                        # Multi-model comparison mode
                        with st.spinner(f"Analyzing {media_type} with all models... This may take a moment."):
                            try:
                                # Run multi-model inference
                                results = inference_multi_model(
                                    temp_file_path, 
                                    user_prompt, 
                                    system_prompt, 
                                    media_type,
                                    max_num_vision_tokens=max_num_vision_tokens,
                                    timestamp_video=timestamp_video,
                                    fps_value=fps_value,
                                    temperature=temperature,
                                    seed=seed
                                )
                                
                                # Store results in session state for full-width display
                                st.session_state.comparison_results = results
                                st.session_state.comparison_filename = uploaded_file.name
                                st.session_state.show_comparison = True
                                
                            except Exception as e:
                                st.error(f"Error during multi-model analysis: {str(e)}")
                                st.info("Please check if the vLLM servers are running and accessible.")
                    else:
                        # Single model mode
                        with st.spinner(f"Analyzing {media_type}... This may take a moment."):
                            try:
                                # Determine model family and use appropriate inference function
                                model_family = MODEL_FAMILY_MAPPING.get(selected_model, "qwen")
                                original_model_name = selected_model.split(": ", 1)[1] if ": " in selected_model else selected_model
                                
                                if model_family == "intern":
                                    # Use Intern inference with appropriate parameters
                                    intern_params = {
                                        "media_path": temp_file_path,
                                        "user_prompt": user_prompt,
                                        "system_prompt": system_prompt,
                                        "model_name": original_model_name,
                                        "media_type": media_type,
                                        "temperature": temperature,
                                        "seed": seed
                                    }
                                        
                                    if 'max_tokens' in locals():
                                        intern_params["max_tokens"] = max_tokens
                                    else:
                                        intern_params["max_tokens"] = 2048
                                        
                                    if 'enable_thinking' in locals():
                                        intern_params["enable_thinking"] = enable_thinking
                                    else:
                                        intern_params["enable_thinking"] = True
                                        
                                    if 'use_tools' in locals():
                                        intern_params["use_tools"] = use_tools
                                    else:
                                        intern_params["use_tools"] = False
                                    
                                    response = intern_inference(**intern_params)
                                else:
                                    # Use Qwen inference
                                    response = qwen_inference(
                                        media_path=temp_file_path,
                                        user_prompt=user_prompt,
                                        system_prompt=system_prompt,
                                        model_name=original_model_name,
                                        media_type=media_type,
                                        max_num_vision_tokens=max_num_vision_tokens,
                                        timestamp_video=timestamp_video,
                                        fps_value=fps_value,
                                        temperature=temperature,
                                        seed=seed
                                    )
                                
                                # Display results
                                st.subheader("🤖 AI Analysis Result")
                                
                                # Show which model was used
                                current_selected = st.session_state.get("selected_model", "cosmos-reason1.1")
                                # Ensure we have a valid model name (not "Compare All Models")
                                if current_selected == "Compare All Models":
                                    # This shouldn't happen in single model mode, but fallback to first model
                                    model_name = list(MODEL_TO_IP_DICT.keys())[0]
                                else:
                                    model_name = current_selected
                                selected_port = MODEL_TO_IP_DICT[model_name]
                                st.markdown(f"**Model:** `{model_name}` | **Endpoint:** `{selected_port}`")
                                
                                st.markdown(response)
                                
                                # Option to download response
                                st.download_button(
                                    label="📥 Download Response",
                                    data=response,
                                    file_name=f"analysis_{uploaded_file.name}.txt",
                                    mime="text/plain"
                                )
                                
                                # Clear any previous comparison results
                                st.session_state.show_comparison = False
                                
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                st.info("Please check if the vLLM server is running and accessible.")
                else:
                    st.warning("Please enter a question before analyzing.")
        
        # Display comparison results outside of columns for full page width
        if st.session_state.get('show_comparison', False) and 'comparison_results' in st.session_state:
            results = st.session_state.comparison_results
            uploaded_filename = st.session_state.comparison_filename
            
            # Add spacing from the columns above
            st.markdown("---")
            
            # Display results from all models using full page width
            st.subheader("🤖 Model Comparison Results")
            
            # Sort model names alphabetically for consistent display
            model_names = sorted(results.keys())
            
            # Display results in a 2x4 grid layout
            models_per_row = 4
            num_rows = (len(model_names) + models_per_row - 1) // models_per_row  # Ceiling division
            
            for row in range(num_rows):
                # Create columns for this row
                cols = st.columns(models_per_row)
                
                # Fill the columns for this row
                for col_idx in range(models_per_row):
                    model_idx = row * models_per_row + col_idx
                    
                    # Check if we have a model for this position
                    if model_idx < len(model_names):
                        model_name = model_names[model_idx]
                        result = results[model_name]
                        
                        with cols[col_idx]:
                            # Show model info with styling
                            ip_port = MODEL_TO_IP_DICT[model_name]
                            st.markdown(f"### 📊 {model_name}")
                            st.markdown(f"**Endpoint:** `{ip_port}`")
                            
                            if result["success"]:
                                st.markdown("✅ **Status:** Success")
                                st.markdown("**Response:**")
                                # Use a container with border for better visual separation
                                with st.container():
                                    st.markdown(result["response"])
                                
                                # Option to download individual response
                                st.download_button(
                                    label=f"📥 Download",
                                    data=result["response"],
                                    file_name=f"analysis_{model_name}_{uploaded_filename}.txt",
                                    mime="text/plain",
                                    key=f"download_{model_name}",
                                    use_container_width=True
                                )
                            else:
                                st.markdown("❌ **Status:** Error")
                                st.error(f"Error: {result['error']}")
                    else:
                        # Empty column if no model for this position
                        with cols[col_idx]:
                            st.empty()
                
                # Add spacing between rows
                if row < num_rows - 1:
                    st.markdown("<br>", unsafe_allow_html=True)
            
            # Add some spacing before the combined download
            st.markdown("---")
            
            # Option to download combined results
            combined_results = ""
            for model_name, result in results.items():
                ip_port = MODEL_TO_IP_DICT[model_name]
                combined_results += f"=== {model_name} ({ip_port}) ===\n"
                if result["success"]:
                    combined_results += f"Status: Success\n\n{result['response']}\n\n"
                else:
                    combined_results += f"Status: Error\nError: {result['error']}\n\n"
                combined_results += "-" * 50 + "\n\n"
            
            st.download_button(
                label="📥 Download All Results",
                data=combined_results,
                file_name=f"comparison_analysis_{uploaded_filename}.txt",
                mime="text/plain"
            )
            
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass  # File cleanup failed, but not critical
            
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        ### 🎬 Welcome to Multi-Model Analysis Tool
        
        This tool supports both **Qwen** and **Intern-S1** models and allows you to:
        - 🤖 **Select between Qwen and Intern models** or compare all models simultaneously
        - 📤 **Upload videos or images** for AI analysis
        - ▶️ **Play and review** your uploaded videos
        - 💬 **Get intelligent responses** to your questions about the media
        - 📊 **Compare responses** from multiple model families side-by-side
        - 💾 **Download analysis results** for future reference
        - ⚙️ **Use model-specific features** (Qwen's timestamp/FPS control, Intern's thinking mode)
        
        **To get started:**
        1. **Choose your analysis mode** in the sidebar:
           - Select a specific Qwen or Intern model for single analysis
           - Choose "Compare All Models" to see responses from all available models
        2. **Upload a video or image file** using the sidebar
        3. **Configure model-specific settings** (different options appear based on your model choice)
        4. **Choose or write a question** about your media
        5. **Click "Analyze Media"** to get AI insights
        
        **Available Models:**
        """)
        
        # Display available models dynamically
        for model_name, ip_port in MODEL_TO_IP_DICT.items():
            st.markdown(f"- **{model_name}:** `{ip_port}`")
        
        st.markdown("""
        **Supported formats:**
        - **Videos:** MP4, AVI, MOV, MKV
        - **Images:** JPG, JPEG, PNG, BMP, WEBP
        """)
        
        # Display some example questions
        with st.expander("💡 Example Questions You Can Ask"):
            st.markdown("""
            - What are the potential safety hazards?
            - Describe what is happening in this video/image.
            - What objects do you see?
            - What actions are being performed?
            - Are there any people in this media?
            - What is the robot doing?
            - Is this demonstration successful?
            - What could be improved in this process?
            """)

if __name__ == "__main__":
    main() 