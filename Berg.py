import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import os
import pytesseract
from groq import Groq
from requests.exceptions import ConnectionError, Timeout, RequestException
from tenacity import retry, wait_exponential, stop_after_attempt

# Set your Gorq AI API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

if not GROQ_API_KEY:
    raise ValueError("Please set your API key.")

class ImageToTextPipeline:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = Groq(api_key=self.api_key)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def get_chat_completion(self, text):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                model="llama3-8b-8192",
            )
            return chat_completion
        except ConnectionError:
            print("Error: Failed to connect to the server. Retrying...")
            raise
        except Timeout:
            print("Error: The request timed out. Retrying...")
            raise
        except RequestException as e:
            print(f"Error: An error occurred. {e}")
            raise

    def extract_text_from_image(self, image_path):
        # Load the image
        img = Image.open(image_path)
        
        # Perform OCR
        text = pytesseract.image_to_string(img)
        
        return text

    def process_image(self, image_path):
        text = self.extract_text_from_image(image_path)
        if text:
            try:
                chat_completion = self.get_chat_completion(text)
                return chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Failed after several retries: {e}")
                return None
        else:
            print("No text found in the image.")
            return None

    def process_text(self, text):
        try:
            chat_completion = self.get_chat_completion(text)
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Failed after several retries: {e}")
            return None

    def save_pipeline(self, filename):
        # Temporarily remove the client before pickling
        client = self.client
        self.client = None
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        # Restore the client
        self.client = client

    @classmethod
    def load_pipeline(cls, filename, api_key):
        with open(filename, 'rb') as f:
            pipeline = pickle.load(f)
        # Reinitialize the client
        pipeline.client = Groq(api_key=api_key)
        return pipeline

# Load the saved pipeline
pipeline = ImageToTextPipeline.load_pipeline('pipeline.pkl', GROQ_API_KEY)

st.title("Berg LPU based Gen Ai")

# Add a text input field
text_input_value = st.text_input("Enter text or upload an image:", key="text_input")

# Add an image upload section
uploaded_image = st.file_uploader("Or upload an image:", type=["jpg", "jpeg", "png"], key="uploaded_image")

if uploaded_image:
    # Convert the uploaded image to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Convert the OpenCV image to a PIL image
    pil_image = Image.fromarray(opencv_image)

    # Save the PIL image to a temporary file
    temp_image_path = "temp.jpg"
    pil_image.save(temp_image_path)

    # Process the image using the pipeline
    output = pipeline.process_image(temp_image_path)

    # Display the output
    st.write("")
    st.write(output)

    # Remove the temporary image file
    os.remove(temp_image_path)

    # Reset the uploaded image
    uploaded_image = None

elif text_input_value:
    # Process the text input using the pipeline
    output = pipeline.process_text(text_input_value)

    # Display the output
    st.write("⚡️")
    st.write("clear the inputs 🌝")
    st.write(output)

    # Reset the text input field
    text_input_value = ""

else:
    st.write("Please enter text or upload an image.")

# Add a reload button
if st.button("Reload"):
    st.session_state.clear()
    st.experimental_rerun()