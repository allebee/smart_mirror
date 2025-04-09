from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
import os
import requests
import pygame
from io import BytesIO

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your API key

# Initialize the ChatOpenAI with gpt-4o model
model = ChatOpenAI(model="gpt-4o")

# Path to your image
image_path = "/Users/user/Desktop/sanamed_mirror/images (1).jpeg"
base64_image = encode_image(image_path)

# Create a message with both text and image
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Опиши кожу лица как дерматолог, основываясь на этом изображении."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]
)

# Get response from the model
response = model.invoke([message])
text_response = response.content
print(text_response)

# Convert the text response to speech
def text_to_speech(text, voice="alloy"):
    """
    Convert text to speech using OpenAI TTS API
    voice options: alloy, echo, fable, onyx, nova, shimmer
    """
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": voice
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.content
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Play the audio
def play_audio(audio_data):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(BytesIO(audio_data))
    sound.play()
    pygame.time.wait(int(sound.get_length() * 1000))  # Wait for the sound to finish playing

# Generate and play TTS
audio_data = text_to_speech(text_response, voice="echo")  # You can change the voice here
if audio_data:
    play_audio(audio_data)