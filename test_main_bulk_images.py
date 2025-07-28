import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

from pipeline import FoodPipeline

# --- Model Config ---
OD_MODEL_PATH = "models/yolov8m-oiv7/train9/weights/best.pt"
CLS_MODEL_PATH = "models/efficientnetv2s/best_model.pth"
CLS_MAPPING_PATH = "class_mapping.json"
NUM_CLASSES = 84

# --- Load Pipeline ---
pipeline = FoodPipeline(
    od_model_path=OD_MODEL_PATH,
    cls_model_path=CLS_MODEL_PATH,
    cls_mapping_path=CLS_MAPPING_PATH,
    num_classes=NUM_CLASSES
)

# --- Directory Configuration ---
SOURCE_DIR = r"dataset\images\test_sample_for_client"  # 🔁 Replace this with your directory
OUTPUT_EXCEL_PATH = "food_predictions.xlsx"

# --- Collect Predictions ---
results = []

def process_image(image_path, class_folder):
    try:
        pred = pipeline.run_inference(
            image_path=image_path,
            od_confidence_thresh=0.3,
            cls_confidence_thresh=0.35,
            filtering_method="advanced",
            visualize=False,
            save_viz_path=None
        )
        detected_items = list(pred.keys()) if pred else []
        return {
            "image_path": image_path,
            "class_name": class_folder,
            "predicted_items": ", ".join(detected_items) if detected_items else "None"
        }
    except Exception as e:
        return {
            "image_path": image_path,
            "class_name": class_folder,
            "predicted_items": f"Error: {e}"
        }

def process_folder(class_folder):
    folder_path = os.path.join(SOURCE_DIR, class_folder)
    if not os.path.isdir(folder_path):
        return []
    folder_results = []
    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(folder_path, image_name)
        folder_results.append(process_image(image_path, class_folder))
    return folder_results

for class_folder in tqdm(os.listdir(SOURCE_DIR), desc="Processing folders"):
    results.extend(process_folder(class_folder))

# --- Save to Excel ---
df = pd.DataFrame(results)
df.to_excel(OUTPUT_EXCEL_PATH, index=False)

print(f"\n✅ Prediction results saved to: {OUTPUT_EXCEL_PATH}")