import streamlit as st
import numpy as np
from PIL import Image

# Import our custom image forensics modules
from src.image.preprocess import preprocess_image
from src.image.detector import ImageForensicsDetector
from src.image.explain import generate_gradcam

# Streamlit Page Configuration
st.set_page_config(page_title="SATYA", page_icon="🛡️")

# Title and Subtitle
st.title("SATYA")
st.subheader("AI Content Forensics")

# Create a modular UI using tabs for future expansion
tabs = st.tabs(["Image Forensics", "Audio Forensics", "Text Forensics"])

# --- IMAGE FORENSICS TAB ---
with tabs[0]:
    st.markdown("### Image Deepfake Detector")
    
    # 1. File uploader accepting JPG, JPEG, PNG, WEBP
    uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        try:
            # 2. Display the uploaded image
            # Replace deprecated use_container_width with width="stretch"
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", width="stretch")
            
            # 3. Initialize detector and attempt to load the model
            # This will raise a FileNotFoundError if the trained model doesn't exist
            detector = ImageForensicsDetector()
            
            # If the model loads successfully, run preprocessing and prediction
            with st.spinner("Analyzing image..."):
                input_tensor = preprocess_image(image)
                result = detector.predict(input_tensor)
                
            # Display results
            st.markdown("### Analysis Results")
            label = result["label"].upper()
            confidence = result["confidence"]
            device_used = result["device"].upper()
            
            # Show REAL / MANIPULATED predictions with confidence
            if label == "FAKE":
                st.error(f"**Prediction:** MANIPULATED ({confidence:.2%} confidence)")
            else:
                st.success(f"**Prediction:** REAL ({confidence:.2%} confidence)")
                
            st.info(f"Inference ran on: {device_used}")
            
            # Display Grad-CAM explanation
            st.markdown("### AI Explanation (Grad-CAM)")
            with st.spinner("Generating explanation..."):
                original_image_np = np.array(image)
                gradcam_img = generate_gradcam(detector.model, input_tensor, original_image_np)
                
                if gradcam_img is not None:
                    st.image(gradcam_img, caption="Grad-CAM Heatmap", width="stretch")
                else:
                    st.warning("Could not generate Grad-CAM visualization.")
                    
        except FileNotFoundError:
            # Handle missing model file explicitly with the requested clean message
            st.info("Image Forensics model is not available yet. Train the SATYA image model first.")
        except Exception as e:
            # Catch all other errors gracefully
            st.error(f"An unexpected error occurred: {e}")

# --- AUDIO FORENSICS TAB ---
with tabs[1]:
    st.info("Audio Forensics module will be added here.")

# --- TEXT FORENSICS TAB ---
with tabs[2]:
    st.info("Text Forensics module will be added here.")

