import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Page Configuration ---
st.set_page_config(
    page_title="Butterfly Species Classifier",
    page_icon="🦋",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Elegant CSS Styling ---

st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    h1 {
        color: #2E4057;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #566573;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 20px;
        border-top: 5px solid #4CAF50;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        color: #1E8449;
    }
    .confidence-text {
        font-size: 18px;
        color: #2C3E50;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title and Header ---
st.markdown("<h1>🦋 Butterfly Image Classification AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload a butterfly image to identify its species instantly using Deep Learning</p>", unsafe_allow_html=True)

# --- Sidebar Info ---
st.sidebar.header("ℹ️ About the App")
st.sidebar.write("This AI application uses a fine-tuned MobileNetV2 architecture trained on thousands of butterfly images to accurately classify species.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Requirements:**")
st.sidebar.code("streamlit\ntensorflow\npillow\nnumpy")

# --- Helper Functions for Caching ---
@st.cache_resource
def load_butterfly_model():
    return tf.keras.models.load_model('butterfly_model.h5')

@st.cache_data
def load_class_names():
    with open('class_names.json', 'r') as f:
        data = json.load(f)

    return {int(k): v for k, v in data.items()}

# --- Load Model & Class Labels ---
try:
    model = load_butterfly_model()
    class_mapping = load_class_names()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model or JSON files. Please check if 'butterfly_model.h5' and 'class_names.json' are in the same folder. Details: {e}")
    model_loaded = False

# --- Main Interface ---
if model_loaded:
    uploaded_file = st.file_uploader("Choose a butterfly image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded View")
        
        with col2:
            st.markdown("### 🧠 Model Prediction")
            with st.spinner("Analyzing the features and patterns... Please wait."):
                
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized, dtype="float32")
                img_array = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_array)
                
               
                predictions = model.predict(img_preprocessed)
                predicted_class_idx = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_class_idx]) * 100
                
                
                species_name = class_mapping.get(predicted_class_idx, f"Unknown Species (ID: {predicted_class_idx})")
                
            
            if confidence > 30:
                st.markdown(f"""
                    <div class='metric-container'>
                        <p style='margin:0; font-size:14px; color:#7F8C8D; text-transform: uppercase;'>Detected Species</p>
                        <p class='result-text'>{species_name}</p>
                        <p class='confidence-text'>Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(int(confidence))
            else:
                st.warning("⚠️ The AI is uncertain about this image. Please upload a clearer image.")
                st.markdown(f"**Top guess:** {species_name} ({confidence:.2f}%)")

else:
    st.info("💡 Please fix the file path errors above to launch the interactive UI.")