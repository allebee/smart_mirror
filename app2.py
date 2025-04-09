import streamlit as st
import cv2
import numpy as np
import base64
from PIL import Image
import io
import time
import openai
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Smart Mirror | Skin Analysis",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# OpenAI API key setup
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

# CSS styles with improved design
def load_css():
    css = """
    <style>
    /* Main styles */
    .main {
        background-color: #0a192f;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1, h2, h3 {
        color: #64ffda;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Container styling */
    .container {
        background-color: rgba(16, 33, 65, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(100, 255, 218, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Camera container */
    .camera-container {
        border: 2px solid #64ffda;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: rgba(16, 33, 65, 0.6);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1a365d;
        color: #64ffda;
        border: 2px solid #64ffda;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }
    
    .stButton>button:hover {
        background-color: #64ffda;
        color: #0a192f;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Results styling */
    .results-container {
        background-color: rgba(16, 33, 65, 0.6);
        border: 1px solid rgba(100, 255, 218, 0.3);
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background-color: rgba(30, 50, 92, 0.5);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #64ffda;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        background-color: rgba(40, 60, 102, 0.6);
    }
    
    .recommendation-category {
        color: #64ffda;
        font-weight: 600;
        margin-bottom: 5px;
        font-size: 1.1rem;
    }
    
    .recommendation-advice {
        color: #ffffff;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Sound wave animation - ENHANCED */
    .sound-wave {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 80px;
        margin: 15px 0;
    }
    
    .sound-wave .bar {
        background: linear-gradient(to top, #64ffda, #805ad5);
        width: 12px;
        margin: 0 5px;
        border-radius: 5px;
        animation: sound-wave-anim 1s ease-in-out infinite;
    }
    
    .sound-wave .bar:nth-child(1) { animation-delay: 0.1s; height: 40px; }
    .sound-wave .bar:nth-child(2) { animation-delay: 0.2s; height: 50px; }
    .sound-wave .bar:nth-child(3) { animation-delay: 0.3s; height: 65px; }
    .sound-wave .bar:nth-child(4) { animation-delay: 0.4s; height: 50px; }
    .sound-wave .bar:nth-child(5) { animation-delay: 0.5s; height: 40px; }
    
    @keyframes sound-wave-anim {
        0%, 100% {
            transform: scaleY(1);
        }
        50% {
            transform: scaleY(0.6);
        }
    }
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 30px 0;
    }
    
    .loading-spinner .spinner {
        width: 50px;
        height: 50px;
        border: 5px solid rgba(100, 255, 218, 0.3);
        border-radius: 50%;
        border-top-color: #64ffda;
        animation: spinner 1s linear infinite;
    }
    
    @keyframes spinner {
        to {
            transform: rotate(360deg);
        }
    }
    
    /* Voice selector styling */
    .voice-selector {
        background-color: rgba(30, 50, 92, 0.5);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    .voice-selector label {
        color: #64ffda;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    /* Skin type badge */
    .skin-type-badge {
        display: inline-block;
        background-color: rgba(100, 255, 218, 0.2);
        color: #64ffda;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        margin-bottom: 15px;
        border: 1px solid rgba(100, 255, 218, 0.3);
    }
    
    /* Summary text */
    .summary-text {
        line-height: 1.6;
        font-size: 1.05rem;
        background-color: rgba(30, 50, 92, 0.4);
        padding: 15px;
        border-radius: 8px;
        border-left: 3px solid rgba(100, 255, 218, 0.5);
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        margin: 10px 0;
    }
    
    /* Input method selector */
    .input-method {
        margin-bottom: 15px;
    }
    
    .input-method .stRadio > div {
        display: flex;
        flex-direction: row;
    }
    
    .input-method .stRadio > div > label {
        background-color: rgba(30, 50, 92, 0.5);
        padding: 10px;
        border-radius: 8px;
        margin-right: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .input-method .stRadio > div > label:hover {
        background-color: rgba(40, 60, 102, 0.6);
    }
    
    /* API key input */
    .api-key-input {
        margin-bottom: 20px;
        background-color: rgba(16, 33, 65, 0.6);
        padding: 15px;
        border-radius: 8px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Enhanced sound wave animation
def create_sound_wave_html(is_playing=False):
    display = "flex" if is_playing else "none"
    html = f"""
    <div class="sound-wave" style="display: {display};">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
    </div>
    """
    return html

# Better loading animation
def create_loading_animation():
    html = """
    <div class="loading-spinner">
        <div class="spinner"></div>
    </div>
    <div style="text-align: center; color: #64ffda; margin-top: 10px;">
        Analyzing your skin...
    </div>
    """
    return html

# Capture from webcam
def capture_from_webcam():
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    img_file_buffer = st.camera_input("Take a photo", key="camera")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        return image
    return None

# Skin analysis function using GPT-4o
def analyze_skin(image, api_key):
    try:
        openai.api_key = api_key
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Request to GPT-4o
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You're a skin analysis expert.
                    Analyze the provided image and give general skin care recommendations.
                    DO NOT provide medical diagnoses. Focus on:
                    1. Skin type (normal, dry, oily, combination)
                    2. Overall skin condition
                    3. 3-4 specific care recommendations
                    Provide the result in the following JSON format:
                    {
                        "summary": "Brief description of skin condition (5-6 sentences)",
                        "skin_type": "Skin type",
                        "recommendations": [
                            {"category": "Category1", "advice": "Recommendation1"},
                            {"category": "Category2", "advice": "Recommendation2"},
                            {"category": "Category3", "advice": "Recommendation3"}
                        ]
                    }
                    in russian
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze my skin and provide care recommendations."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        
        # Extract result
        result = response.choices[0].message.content
        return result
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None

# Text to speech function
def text_to_speech(text, voice_type, api_key):
    try:
        openai.api_key = api_key
        
        # Request to API
        response = openai.audio.speech.create(
            model="tts-1",
            voice=voice_type,
            input=text
        )
        
        # Save audio to temporary file
        audio_file = "temp_audio.mp3"
        response.stream_to_file(audio_file)
        
        # Read file for browser playback
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        
        return audio_bytes
    except Exception as e:
        st.error(f"Speech generation error: {str(e)}")
        return None

# Improved recommendations display
def display_recommendations(recommendations):
    for rec in recommendations:
        html = f"""
        <div class="recommendation-card">
            <div class="recommendation-category">{rec["category"]}</div>
            <div class="recommendation-advice">{rec["advice"]}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

# Main app function
def main():
    load_css()
    
    # Container for the entire app
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Centered title
    st.markdown("<h1 style='text-align: center;'>Smart Mirror with Skin Analysis</h1>", unsafe_allow_html=True)
    
    # API key request
    if not st.session_state.get('openai_api_key'):
        st.markdown('<div class="api-key-input">', unsafe_allow_html=True)
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        st.markdown('</div>', unsafe_allow_html=True)
        if api_key:
            st.session_state.openai_api_key = api_key
    
    # Check for API key
    if not st.session_state.get('openai_api_key'):
        st.warning("Please enter your OpenAI API key to use the application.")
        st.markdown('</div>', unsafe_allow_html=True)  # Close container
        return
    
    # Initialize session state
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'audio' not in st.session_state:
        st.session_state.audio = None
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    
    # Create two columns for interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h2>Camera</h2>", unsafe_allow_html=True)
        
        # Input method selection
        st.markdown('<div class="input-method">', unsafe_allow_html=True)
        input_method = st.radio("Choose input method:", ["Camera", "Upload photo"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if input_method == "Camera":
            # Capture image from camera
            image = capture_from_webcam()
            if image is not None:
                st.session_state.image = image
        else:
            # Upload image
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                st.session_state.image = Image.open(uploaded_file)
                # Display uploaded image
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                st.image(st.session_state.image, caption="Uploaded image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        if st.session_state.image is not None:
            if st.button("Analyze"):
                with st.spinner():
                    st.markdown(create_loading_animation(), unsafe_allow_html=True)
                    
                    # Analyze image
                    result_json = analyze_skin(st.session_state.image, st.session_state.openai_api_key)
                    
                    if result_json:
                        st.session_state.analysis_result = json.loads(result_json)
                        
                        # Generate audio
                        voice_type = st.session_state.get('voice_type', 'alloy')
                        audio = text_to_speech(st.session_state.analysis_result["summary"], voice_type, st.session_state.openai_api_key)
                        if audio:
                            st.session_state.audio = audio
                            st.session_state.is_playing = True
                        
                        # Force refresh to display results
                        st.rerun()
    
    with col2:
        st.markdown("<h2>Analysis Results</h2>", unsafe_allow_html=True)
        
        # Voice settings in a nice container
        st.markdown('<div class="voice-selector">', unsafe_allow_html=True)
        voice_options = {
            "alloy": "Alloy (neutral)",
            "echo": "Echo (deep)",
            "fable": "Fable (soft)",
            "onyx": "Onyx (strong)",
            "nova": "Nova (female)",
            "shimmer": "Shimmer (positive)"
        }
        selected_voice = st.selectbox(
            "Select voice for audio:",
            options=list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=0
        )
        st.session_state.voice_type = selected_voice
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results
        if st.session_state.analysis_result is not None:
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            result = st.session_state.analysis_result
            
            # Display skin type with badge
            st.markdown(f'<div class="skin-type-badge">Skin Type: {result["skin_type"]}</div>', unsafe_allow_html=True)
            
            # Audio and visualization - auto-play
            if st.session_state.audio is not None:
                # Custom audio component for auto-play
                audio_html = f"""
                <audio autoplay controls>
                    <source src="data:audio/mp3;base64,{base64.b64encode(st.session_state.audio).decode()}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
                
                # Prominent sound wave animation
                st.markdown(create_sound_wave_html(True), unsafe_allow_html=True)
                
                # Summary with better styling
                st.markdown("<h3>Overall Assessment</h3>", unsafe_allow_html=True)
                st.markdown(f'<div class="summary-text">{result["summary"]}</div>', unsafe_allow_html=True)
            
            # Improved recommendations display
            st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
            display_recommendations(result["recommendations"])
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset button
        if st.session_state.analysis_result is not None:
            if st.button("Start Over"):
                # Reset state
                st.session_state.image = None
                st.session_state.analysis_result = None
                st.session_state.audio = None
                st.session_state.is_playing = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close container

if __name__ == "__main__":
    main()