
import os
import json
import time

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, UnidentifiedImageError


# =====================================================
# 1. APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CropNet-CNN | Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    APP_DIR,
    "CropNet_CNN_Best.keras"
)

MAPPING_PATH = os.path.join(
    APP_DIR,
    "class_mapping.json"
)

IMAGE_SIZE = (224, 224)

MAX_FILE_SIZE = 10 * 1024 * 1024

CONFIDENCE_THRESHOLD = 0.60

ALLOWED_EXTENSIONS = [
    "jpg",
    "jpeg",
    "png",
    "webp"
]


# =====================================================
# 2. PAGE DESIGN
# =====================================================

st.title("🌿 CropNet-CNN")

st.subheader(
    "Automated Crop Leaf Disease Detection System"
)

st.write(
    "Upload a crop leaf image to identify its "
    "predicted crop species and leaf disease."
)

st.info(
    "AI-powered leaf disease screening using "
    "a trained Convolutional Neural Network."
)

st.divider()


# =====================================================
# 3. LOAD CLASS MAPPING
# =====================================================

@st.cache_data
def load_class_names():

    with open(MAPPING_PATH, "r") as file:

        mapping = json.load(file)

    if isinstance(mapping, list):

        class_names = mapping

    elif isinstance(mapping, dict):

        # Supports index-to-class mappings
        if all(str(k).isdigit() for k in mapping):

            class_names = [
                mapping[k]
                for k in sorted(
                    mapping,
                    key=lambda x: int(x)
                )
            ]

        # Supports class-to-index mappings
        elif all(
            isinstance(v, int)
            for v in mapping.values()
        ):

            class_names = [
                name
                for name, index in sorted(
                    mapping.items(),
                    key=lambda item: item[1]
                )
            ]

        else:

            raise ValueError(
                "Unsupported class mapping format."
            )

    else:

        raise ValueError(
            "Invalid class mapping format."
        )

    if len(class_names) != 15:

        raise ValueError(
            f"Expected 15 classes, found {len(class_names)}."
        )

    return [str(name) for name in class_names]


# =====================================================
# 4. LOAD TRAINED CNN MODEL
# =====================================================

@st.cache_resource
def load_cropnet_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Trained CropNet-CNN model not found."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if model.input_shape != (None, 224, 224, 3):

        raise ValueError(
            f"Unexpected input shape: {model.input_shape}"
        )

    if model.output_shape[-1] != 15:

        raise ValueError(
            "Model must generate 15 class probabilities."
        )

    return model


try:

    class_names = load_class_names()

    model = load_cropnet_model()

    st.success(
        "CropNet-CNN model loaded successfully!"
    )

except Exception as error:

    st.error(
        f"Application initialization failed: {error}"
    )

    st.stop()


# =====================================================
# 5. IMAGE PREPROCESSING FUNCTION
# =====================================================

def preprocess_leaf_image(uploaded_file):

    uploaded_file.seek(0)

    image = Image.open(
        uploaded_file
    )

    image.load()

    if image.width <= 0 or image.height <= 0:

        raise ValueError(
            "Image dimensions must be greater than zero."
        )

    # Convert uploaded image to RGB
    image = image.convert("RGB")

    # Bilinear resizing
    resized_image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # Convert pixels to float32 and normalize
    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    ) / 255.0

    # Add batch dimension
    image_tensor = np.expand_dims(
        image_array,
        axis=0
    )

    return image, resized_image, image_tensor


# =====================================================
# 6. FILE UPLOAD SECTION
# =====================================================

st.header("📤 Upload Crop Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=ALLOWED_EXTENSIONS,
    help="Supported: JPG, JPEG, PNG and WEBP. Maximum size: 10 MB."
)


# =====================================================
# 7. VALIDATE AND PROCESS UPLOADED IMAGE
# =====================================================

if uploaded_file is not None:

    file_size = uploaded_file.size

    if file_size <= 0:

        st.error("Uploaded file is empty.")

        st.stop()

    if file_size > MAX_FILE_SIZE:

        st.error(
            "File size exceeds the maximum limit of 10 MB."
        )

        st.stop()

    try:

        original_image, processed_image, image_tensor = (
            preprocess_leaf_image(uploaded_file)
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as error:

        st.error(
            f"Invalid or corrupted image: {error}"
        )

        st.stop()

    st.success("Image validation successful!")

    st.write(
        f"File size: {file_size / (1024 * 1024):.2f} MB"
    )

    st.write(
        f"Original dimensions: "
        f"{original_image.width} × {original_image.height}"
    )

    st.write(
        f"Processed tensor shape: {image_tensor.shape}"
    )

    st.write(
        f"Pixel range: "
        f"{image_tensor.min():.3f} to "
        f"{image_tensor.max():.3f}"
    )

    # =================================================
    # 8. DISPLAY UPLOADED IMAGE
    # =================================================

    st.divider()

    st.header("🖼️ Uploaded Leaf Image")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            original_image,
            caption="Original Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.image(
            processed_image,
            caption="224 × 224 Processed Image",
            use_container_width=True
        )

    # =================================================
    # 9. MODEL INFERENCE
    # =================================================

    st.divider()

    st.header("🔍 Disease Detection Result")

    with st.spinner(
        "Analyzing leaf image using CropNet-CNN..."
    ):

        start_time = time.perf_counter()

        probabilities = model(
            image_tensor,
            training=False
        ).numpy()[0]

        inference_time = (
            time.perf_counter() - start_time
        )

    if not np.all(np.isfinite(probabilities)):

        st.error(
            "Model generated invalid probability values."
        )

        st.stop()

    if not np.isclose(
        np.sum(probabilities),
        1.0,
        atol=1e-4
    ):

        st.error(
            "Invalid Softmax probability distribution."
        )

        st.stop()

    # =================================================
    # 10. TOP-1 PREDICTION
    # =================================================

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    confidence_percentage = confidence * 100

    # =================================================
    # 11. PARSE CROP AND DISEASE NAME
    # =================================================

    if "___" in predicted_class:

        crop_name, disease_name = (
            predicted_class.split("___", 1)
        )

    else:

        crop_name = "Not specified"

        disease_name = predicted_class

    crop_name = crop_name.replace("_", " ")

    disease_name = disease_name.replace("_", " ")

    is_healthy = (
        disease_name.strip().lower() == "healthy"
    )

    # =================================================
    # 12. DISPLAY DIAGNOSTIC RESULT
    # =================================================

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Crop",
            crop_name
        )

    with result_col2:

        st.metric(
            "Confidence",
            f"{confidence_percentage:.2f}%"
        )

    st.subheader(
        "Predicted Condition"
    )

    st.write(
        f"**{disease_name}**"
    )

    # =================================================
    # 13. LOW CONFIDENCE WARNING
    # =================================================

    if confidence < CONFIDENCE_THRESHOLD:

        st.warning(
            "⚠️ Uncertain / Out-of-Distribution. "
            "The model confidence is below 60%. "
            "Please capture a clearer leaf image "
            "and try again."
        )

    elif is_healthy:

        st.success(
            "The model predicts a healthy leaf."
        )

    else:

        st.success(
            "The model predicts a possible leaf disease."
        )

    # =================================================
    # 14. TOP-3 PREDICTIONS
    # =================================================

    st.divider()

    st.header("🏆 Top-3 Predictions")

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        class_name = class_names[index]

        probability = float(
            probabilities[index]
        )

        percentage = probability * 100

        st.write(
            f"**{rank}. {class_name.replace('_', ' ')}**"
        )

        st.progress(
            probability,
            text=f"{percentage:.2f}%"
        )

    # =================================================
    # 15. INFERENCE INFORMATION
    # =================================================

    st.divider()

    st.subheader("⚙️ Inference Information")

    st.write(
        f"Model: CropNet-CNN"
    )

    st.write(
        f"Input Tensor: {image_tensor.shape}"
    )

    st.write(
        f"Number of Classes: {len(class_names)}"
    )

    st.write(
        f"Inference Time: {inference_time:.5f} seconds"
    )

    st.caption(
        "This application provides AI-based preliminary "
        "screening. Predictions should not replace "
        "professional agricultural diagnosis."
    )

else:

    st.info(
        "Please upload a crop leaf image to start detection."
    )


# =====================================================
# 16. FOOTER
# =====================================================

st.divider()

st.caption(
    "CropNet-CNN | Automated Crop Leaf Disease Detection System"
)

st.caption(
    "Academic Project – Neural Network & Deep Learning"
)
