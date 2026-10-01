import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
MODEL_PATH = "car_damage.keras"
CAR_MODEL_PATH = "car_classifier.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["01-minor", "02-moderate", "03-severe"]
CAR_CLASS_NAMES = ["car", "not_car"]

# ---------------------------------------------------------
# Page title
# ---------------------------------------------------------
st.title("Car Damage Severity Verifier")
st.write("Upload a photo of a damaged car and click Predict to see the severity.")

# ---------------------------------------------------------
# Load the models once (cached so they don't reload on every click)
# ---------------------------------------------------------
@st.cache_resource
def load_severity_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_car_model():
    return tf.keras.models.load_model(CAR_MODEL_PATH)

model = load_severity_model()
car_model = load_car_model()

# ---------------------------------------------------------
# Let the user upload an image
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Choose a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded photo")

    if st.button("Predict Severity"):
        # Prepare the image the same way the models were trained on
        img_resized = image.resize(IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

        # Step 1: check if it's actually a car
        car_predictions = car_model.predict(img_array, verbose=0)[0]
        car_predicted_class = CAR_CLASS_NAMES[np.argmax(car_predictions)]

        if car_predicted_class == "not_car":
            st.warning("This doesn't look like a car. Please upload a photo of a car.")
        else:
            # Step 2: predict damage severity
            predictions = model.predict(img_array, verbose=0)[0]
            predicted_class = CLASS_NAMES[np.argmax(predictions)]

            st.write("Predicted severity:", predicted_class)

            st.write("Confidence per class:")
            for name, score in zip(CLASS_NAMES, predictions):
                st.write(f"{name}: {score * 100:.1f}%")
