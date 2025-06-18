# food_pipeline/pipeline.py
import torch
import torch.nn as nn
import json
from ultralytics import YOLO
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from PIL import Image
from typing import List, Dict, Any

from utils.calorie_database import get_calories

def is_box_inside(inner_box: List[int], outer_box: List[int]) -> bool:
    """Checks if the inner_box is completely contained within the outer_box."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

def filter_nested_boxes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters out bounding boxes that are nested inside others."""
    if not detections:
        return []

    boxes = [d['box'] for d in detections]
    is_inner = [False] * len(boxes)

    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(boxes):
            if i == j:
                continue
            if is_box_inside(box1, box2):
                is_inner[i] = True
                break

    return [detections[i] for i, is_in in enumerate(is_inner) if not is_in]

class FoodPipeline:
    def __init__(self, od_model_path: str, cls_model_path: str, cls_mapping_path: str, num_classes: int):
        """
        Initializes the pipeline by loading both object detection and classification models.
        """
        print("Initializing the Food Inference Pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # load object detection model
        self.od_model = YOLO(od_model_path)
        self.od_model.to(self.device)
        
        # load classification model
        self._load_classification_model(cls_model_path, num_classes)

        # Load Classification Class Mapping
        with open(cls_mapping_path, 'r') as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Define classes that trigger the second-stage classification
        self.generic_classes_for_classification = {"food", "drink", "plate", "bowl", "dessert", "fast food", 
                                                   "coffee cup", "baked goods", "dairy product", "hamburger", 
                                                   "mixing bowl", "mug", "salad", "sandwich", "serving tray",
                                                   "taco", "wine glass"}
        
        print("Pipeline initialized successfully.")

    def _load_classification_model(self, model_path: str, num_classes: int):
        """Helper method to load the EfficientNet model and its weights."""
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.cls_model = efficientnet_v2_s(weights=weights)
        
        # Rebuild the classifier head to match the trained model
        num_ftrs = self.cls_model.classifier[1].in_features
        self.cls_model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.37, inplace=True),
            nn.Linear(num_ftrs, num_classes),
        )
        
        # Load the fine-tuned weights
        self.cls_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.cls_model.to(self.device)
        self.cls_model.eval()
        
        # Store the required preprocessing transforms
        self.cls_preprocess = weights.transforms()

    def _classify_cropped_image(self, cropped_image: Image.Image) -> tuple[str, float]:
        """
        Runs REAL inference with the classification model on a cropped image.
        """
        # Apply the correct transformations and move to device
        input_tensor = self.cls_preprocess(cropped_image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.cls_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Map index to class name
        class_name = self.idx_to_class.get(predicted_idx.item(), "Unknown")
        return class_name, confidence.item()
    
    def _get_object_detection_results(self, image_path: str, od_confidence_thresh: float) -> List[Dict[str, Any]]:
        results = self.od_model.predict(image_path, conf=od_confidence_thresh, verbose=False)
        result = results[0]
        
        all_detections = []
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = self.od_model.names[class_id]
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            all_detections.append({'class_name': class_name, 'box': coords})

        print(f"Found {len(all_detections)} initial objects.")

        # --- Filtering Step: Remove Nested Bounding Boxes ---
        outer_detections = filter_nested_boxes(all_detections)
        print(f"Found {len(outer_detections)} outermost objects after filtering.")

        return outer_detections
    
    def _get_final_items(self, image_path: str, main_boxes, cls_confidence_thresh: float):
       
        final_items = []
        original_image = Image.open(image_path).convert("RGB")

        for detection in main_boxes:
            class_name = detection['class_name']
            
            # If the OD detection is specific, use it directly
            if class_name.lower() not in self.generic_classes_for_classification:
                print(f"- Found OD item: '{class_name}'. Adding directly.")
                final_items.append(class_name)
            else:
                # Otherwise, crop the box and pass it to the classifier
                print(f"- Found matching OD item: '{class_name}'. Cropping for classification...")
                x1, y1, x2, y2 = detection['box']
                cropped_image = original_image.crop((x1, y1, x2, y2))
                
                specific_item_name, confidence = self._classify_cropped_image(cropped_image)
                
                # Only accept the classification if confidence is high enough
                if confidence >= cls_confidence_thresh:
                    print(f"  -> Classified as '{specific_item_name}' with confidence {confidence:.2f}. ACCEPTED.")
                    final_items.append(specific_item_name)
                else:
                    # If confidence is low, fall back to the generic OD label
                    print(f"  -> Classified as '{specific_item_name}' with confidence {confidence:.2f}. REJECTED.")
                    # final_items.append(class_name)
        return final_items

    def run_inference(self, image_path: str, od_confidence_thresh=0.25, cls_confidence_thresh=0.5) -> Dict[str, str]:
        """
        Executes the full two-stage pipeline on a single image.
        """
        print(f"\n--- Starting inference for {image_path} ---")

        # --- Stage 1: Object Detection ---
        main_boxes = self._get_object_detection_results(image_path, od_confidence_thresh)   

        # --- Stage 2: Process Outermost Boxes ---
        final_items = self._get_final_items(image_path, main_boxes, cls_confidence_thresh)
        
        # --- Stage 3: Calorie Calculation ---
        print("\n--- Final Results and Calorie Calculation ---")
        calorie_summary = {}
        if not final_items:
            print("No food items were found for calorie calculation.")
        else:
            # Use a dictionary to count occurrences of each item
            item_counts = {}
            for item in final_items:
                item_counts[item] = item_counts.get(item, 0) + 1
            
            for item, count in item_counts.items():
                calories = get_calories(item)
                # Add count to the key if more than one instance is found
                display_key = f"{item} (x{count})" if count > 1 else item
                calorie_summary[display_key] = calories
                print(f"Item: {display_key}, Calories: {calories}")
        
        return calorie_summary

