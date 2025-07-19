# streamlit_app.py
import streamlit as st
import tempfile
import os
from PIL import Image
import json
import io

from pipeline import FoodPipeline

# --- App Configuration ---
OD_MODEL_PATH = "models/yolov8m-oiv7/train4/weights/best.pt"
CLS_MODEL_PATH = "models/efficientnetv2s/best_model.pth"
CLS_MAPPING_PATH = "class_mapping.json"
NUM_CLASSES = 64

# --- Preload Pipeline Once ---
@st.cache_resource(show_spinner=False)
def load_pipeline():
    # Check model files exist
    required_files = [OD_MODEL_PATH, CLS_MODEL_PATH, CLS_MAPPING_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found at: {file_path}")

    return FoodPipeline(
        od_model_path=OD_MODEL_PATH,
        cls_model_path=CLS_MODEL_PATH,
        cls_mapping_path=CLS_MAPPING_PATH,
        num_classes=NUM_CLASSES
    )

pipeline = load_pipeline()

# --- Streamlit UI ---
st.set_page_config(page_title="Food Image Nutrition Analyzer", layout="centered")
st.title("🥗 Food Image Nutrition Analyzer")
st.markdown("Upload an image of food to detect items and view their nutritional details.")

uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

import io

if uploaded_file:
    uploaded_file_bytes = uploaded_file.read()

    # Load and show resized image
    image = Image.open(io.BytesIO(uploaded_file_bytes))
    resized_image = image.resize((min(image.width, 400), min(image.height, 400)))
    st.image(resized_image, caption="Uploaded Image", use_container_width=False)

    # Save to temp file for OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file_bytes)
        tmp_path = tmp_file.name

    if st.button("🔍 Analyze"):
        with st.spinner("Running food detection and nutrient analysis..."):
            try:
                results = pipeline.run_inference(
                    image_path=tmp_path,
                    od_confidence_thresh=0.3,
                    cls_confidence_thresh=0.35,
                    filtering_method="advanced",
                    visualize=False,
                    save_viz_path=None
                )
                os.remove(tmp_path)  # Clean up temp file
            except Exception as e:
                st.error(f"Error while processing the image: {e}")
                st.stop()

        if not results:
            st.warning("No recognizable food items were detected.")
        else:
            st.success("Detected food items with nutrition info:")
            for item, nutrient_list in results.items():
                st.subheader(f"🍴 {item}")
                for entry in nutrient_list:
                    st.markdown(f"**{entry['description']}**")
                    for nutrient, val in entry["nutrients"].items():
                        st.write(f"{nutrient}: {val['value']} {val['unit']}")

