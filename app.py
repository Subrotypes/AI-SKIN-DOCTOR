import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ==========================================
# Load environment variables securely
# ==========================================
load_dotenv()  # ensure your .env is in the same folder as app.py
API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================
# Load your custom AI model
# ==========================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_model.keras')

model = load_model()

# ==========================================
# Streamlit UI
# ==========================================
st.title("Skin Infection Detector & AI Analyst 🩺")
st.write("Upload an image. Our custom AI will check for infections, and Gemini will provide a detailed medical description.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # ==========================================
    # PART 1: Custom CNN Model Prediction
    # ==========================================
    st.write("### 1. Custom AI Diagnosis...")

    img_rgb = image.convert('RGB')
    img_resized = img_rgb.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    st.write(f"**Raw Score:** {score:.4f} *(Closer to 0 = Infected, Closer to 1 = Normal)*")
    if score > 0.5:
        confidence = score * 100
        st.success(f"Status: NO PROBLEM. Skin looks fine. ✅\n\n**Confidence:** {confidence:.2f}%")
    else:
        confidence = (1 - score) * 100
        st.error(f"Status: INFECTION DETECTED 🚨\n\n**Confidence:** {confidence:.2f}%")

    st.write("---")

    # ==========================================
    # PART 2: Gemini Detailed Analysis
   # ==========================================
# ==========================================
    # ==========================================
# PART 2: GEMINI DETAILED DESCRIPTION (Safe Version)
# ==========================================
st.write("### 2. Gemini Detailed Analysis 🤖")

if API_KEY:
    try:
        # Configure Gemini with your API key
        genai.configure(api_key=API_KEY)
        
        # List all available models
        available_models = [m.name for m in genai.list_models()]
        

        # Choose a valid model (pick the first one that supports 'generate_content')
        # You can manually pick the model you want if you prefer
        valid_model_name = None
        for m in available_models:
            if "gemini" in m.lower():  # simple filter, adjust if needed
                valid_model_name = m
                break
        
        if valid_model_name is None:
            st.error("No compatible Gemini model found for generate_content.")
        else:
            gemini_model = genai.GenerativeModel(valid_model_name)
            
            with st.spinner(f"Gemini ({valid_model_name}) is analyzing the image..."):
                prompt = (
                    "You are an AI medical assistant. Describe the visual condition of the skin in this image "
                    "in detail. What visual symptoms do you see? Include a standard medical disclaimer at the end."
                )

                # Send both the prompt and the PIL image to Gemini
                response = gemini_model.generate_content([prompt, image])
                
                st.write(response.text)

    except Exception as e:
        st.error(f"Error communicating with Gemini: {e}")
else:
    st.error("🚨 API Key not found! Please make sure you created the .env file and named the variable exactly GEMINI_API_KEY.")