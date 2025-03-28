import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "model_weights.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (Ensure this matches your training labels)
class_names = ['Electrical', 'General contractor Roofing GR4 interior Renovation IR4', 'HVAC', 'Pest Control', 'Plumbing']  # Update as needed

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    image = image.convert("RGB")  # Ensure image is RGB (removes alpha if present)
    image = image.resize((256, 256))  # Resize to match training size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dims to match batch size
    return image

def predict_image(image):
    """Predict the class of an uploaded image."""
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        return class_names[predicted_class], confidence
    except Exception as e:
        return f"Error: {e}", 0.0

# Streamlit UI
st.title("Image Classification App")
st.write("Upload an image to classify it using the trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Perform prediction
    predicted_label, confidence = predict_image(image)
    
    if confidence > 0.6:
        st.success(f"**REAL/FAKE:** REAL")
        st.write(f"**Predicted Label:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    elif confidence > 0:
        st.error(f"**REAL/FAKE:** FAKE")
    else:
        st.warning(f"Prediction Error: {predicted_label}")  # Shows error message if any
