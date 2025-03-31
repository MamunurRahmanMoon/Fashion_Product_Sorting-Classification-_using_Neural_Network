import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

# Load the trained model
MODEL_PATH = "fashion_mnist_model.h5"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def preprocess_image(image):
    size = (28, 28)  # Model expects 28x28 grayscale images

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Enhance contrast (reduce factor to avoid over-darkening)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)

    # Resize with anti-aliasing
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Normalize pixel values
    image_array = np.asarray(image) / 255.0

    # Add batch and channel dimensions
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


# Streamlit app
st.title("Fashion Product Classification by Md Mamunur Rahman Moon")
st.write("Upload an image of a fashion product to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Preprocess the image
    processed_image = preprocess_image(image)
    processed_image_display = (processed_image[0, :, :, 0] * 255).astype(np.uint8)
    processed_image_pil = Image.fromarray(processed_image_display)

    # Display uploaded and processed images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(processed_image_pil, caption="Processed Image", use_column_width=True)

    if st.button("Classify"):
        st.write("Classifying...")

        # Display processed image details
        st.write("### Processed Image Details:")
        st.write(f"Dimensions: {processed_image.shape[1]}x{processed_image.shape[2]}")
        st.write(f"Pixel Count: {processed_image.size}")

        # Provide download option for processed image
        processed_image_pil.save("processed_image.png")
        with open("processed_image.png", "rb") as file:
            btn = st.download_button(
                label="Download Processed Image",
                data=file,
                file_name="processed_image.png",
                mime="image/png",
            )

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display result
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
