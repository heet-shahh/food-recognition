import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from PIL import Image
import json
import os # Added for path joining

# --- Configuration ---
# It's good practice to define paths and filenames at the top
MODEL_PATH = 'models/best_model.pth'
CLASS_MAPPING_PATH = 'class_mapping.json'

# --- Model Loading ---
@st.cache_resource
def load_model():
    """
    Loads the pre-trained EfficientNetV2-S model with a custom classifier.
    The model state dictionary is loaded from the specified MODEL_PATH.
    Caches the model to avoid reloading on every interaction.
    """
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)
    
    # Modify the classifier head for the number of food classes (66 in this case)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.37, inplace=True), # Dropout for regularization
        nn.Linear(num_ftrs, 66),         # Linear layer for 66 classes
    )
    
    # Load the trained weights. Ensure the model file is accessible.
    # map_location=torch.device('cpu') ensures model loads on CPU if no GPU is available
    # or if the model was trained on GPU but inference is on CPU.
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the path is correct.")
        return None, None
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
    
    # Get the preprocessing transforms associated with the weights
    preprocess = weights.transforms()
    return model, preprocess

# --- Class Mapping Loading ---
@st.cache_data
def load_mapping():
    """
    Loads the class mapping from a JSON file.
    The mapping converts class indices (output by the model) to human-readable class names.
    Caches the mapping to avoid reloading on every interaction.
    """
    if not os.path.exists(CLASS_MAPPING_PATH):
        st.error(f"Class mapping file not found at {CLASS_MAPPING_PATH}. Please ensure the path is correct.")
        return None

    with open(CLASS_MAPPING_PATH, 'r') as f:
        class_to_idx = json.load(f)
    
    # Invert the mapping to get idx_to_class
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

# --- Prediction Logic ---
def predict_image(img_pil, model, preprocess, idx_to_class):
    """
    Performs inference on the uploaded image.
    Args:
        img_pil (PIL.Image): The image uploaded by the user.
        model (torch.nn.Module): The loaded PyTorch model.
        preprocess (callable): The preprocessing transform for the model.
        idx_to_class (dict): Mapping from class index to class name.
    Returns:
        tuple: (predicted_label, confidence_score)
               Returns ("Error in prediction", 0.0) if an issue occurs.
    """
    if model is None or preprocess is None or idx_to_class is None:
        return "Model or mapping not loaded", 0.0

    try:
        # Preprocess the image and add a batch dimension
        image_tensor = preprocess(img_pil).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad(): # Disable gradient calculations for inference
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1) # Convert logits to probabilities
            confidence, idx = torch.max(probabilities, 1) # Get the max probability and its index
            
        predicted_class_name = idx_to_class.get(idx.item(), "Unknown class")
        return predicted_class_name, confidence.item()
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Error in prediction", 0.0

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Food Classifier", page_icon="🍽️")

st.title("🍽️ Food Classifier")
st.caption("Upload an image of food, and the model will attempt to classify it into one of 66 categories.")

# Load model and mapping once
model, preprocess = load_model()
idx_to_class = load_mapping()

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image file (JPG, JPEG, or PNG) for classification."
)

if uploaded_file is not None:
    try:
        img_pil = Image.open(uploaded_file).convert("RGB") # Ensure image is in RGB
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_pil, caption="Uploaded Image", use_container_width=True)

        if model and preprocess and idx_to_class:
            # Perform prediction
            label, confidence = predict_image(img_pil, model, preprocess, idx_to_class)
            
            with col2:
                st.markdown(f"### 🍲 Prediction: `{label}`")
                st.markdown(f"### Confidence: `{confidence:.2%}`")
                
                # Display confidence bar
                st.progress(confidence)
                
                if confidence < 0.5 and label != "Error in prediction": # Example threshold
                    st.warning("The model's confidence is a bit low. The prediction might not be accurate.")
        else:
            st.error("Model or class mapping could not be loaded. Please check the file paths and try again.")
            
    except Exception as e:
        st.error(f"An error occurred processing the image: {e}")

else:
    st.info("Please upload an image to see the classification.")

st.markdown("---")
st.markdown("Developed with Streamlit and PyTorch.")
