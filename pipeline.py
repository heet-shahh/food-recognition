# food_pipeline/pipeline.py

import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple

from utils.calorie_database import get_calories

def is_box_inside(inner_box: List[int], outer_box: List[int]) -> bool:
    """
    Checks if the inner_box is completely contained within the outer_box.
    Boxes are in [x1, y1, x2, y2] format.
    """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

def filter_nested_boxes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters out bounding boxes that are nested inside other bounding boxes.
    Only returns the outermost boxes.

    Args:
        detections: A list of detection dictionaries, each with a 'box' key.

    Returns:
        A new list containing only the outermost detections.
    """
    if not detections:
        return []

    boxes = [d['box'] for d in detections]
    is_inner = [False] * len(boxes)

    # Compare every box with every other box
    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(boxes):
            if i == j:
                continue
            # If box1 is inside box2, mark it as an inner box
            if is_box_inside(box1, box2):
                is_inner[i] = True
                break # No need to check other boxes for this inner box

    # Collect all boxes that are not marked as inner
    outermost_detections = [detections[i] for i, is_in in enumerate(is_inner) if not is_in]
    
    return outermost_detections


class FoodPipeline:
    def __init__(self, od_model_path: str, classification_model_path: str):
        """
        Initializes the pipeline by loading the object detection and classification models.
        """
        print("Initializing the Food Inference Pipeline...")
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 1. Load Object Detection Model (YOLO)
        print(f"Loading Object Detection model from: {od_model_path}")
        self.od_model = YOLO(od_model_path)
        self.od_model.to(self.device)

        # 2. Load Image Classification Model
        print(f"Loading Classification model from: {classification_model_path}")
        # Assuming a standard PyTorch model saved with torch.save()
        # In a real scenario, you'd also need to load the model's architecture
        # For demonstration, we'll assume it's a simple model structure.
        self.classification_model = torch.load(classification_model_path, map_location=self.device)
        self.classification_model.eval()
        # You would also need your classification model's class names and transforms
        # For this example, we'll mock the classification output.
        self.kuwaiti_food_classes = ['Balaleet', 'Majboos_Dajaj', 'Jireesh', 'Samosa', 'Warak_Enab', 'Kebab']

        # 3. Define classes that trigger the second-stage classification
        self.generic_classes_for_classification = {"food", "drink", "plate", "bowl", "dessert"}
        print("Pipeline initialized successfully.")

    def _classify_cropped_image(self, cropped_image: Image.Image) -> str:
        """
        Runs inference with the classification model on a cropped image.
        
        NOTE: This is a placeholder/mock function. In a real implementation, you would:
        1. Apply the correct transformations (resize, normalize, ToTensor) to the cropped_image.
        2. Pass the transformed tensor through self.classification_model.
        3. Get the predicted class index and map it to the class name.
        """
        print("  -> (Mock) Running classification on cropped image...")
        # Mocking the classification logic for this example
        # In a real implementation, this would be:
        # transform = ...
        # input_tensor = transform(cropped_image).unsqueeze(0).to(self.device)
        # with torch.no_grad():
        #     output = self.classification_model(input_tensor)
        #     _, predicted_idx = torch.max(output, 1)
        # return self.kuwaiti_food_classes[predicted_idx.item()]
        
        # For now, we return a random mock result
        import random
        return random.choice(self.kuwaiti_food_classes)

    def run_inference(self, image_path: str) -> Dict[str, str]:
        """
        Executes the full two-stage pipeline on a single image.

        Returns:
            A dictionary of detected items and their calorie counts.
        """
        print(f"\n--- Starting inference for {image_path} ---")
        
        # --- Stage 1: Object Detection ---
        results = self.od_model.predict(image_path, verbose=False)
        result = results[0] # Get results for the first image
        
        # Prepare a list of detections for filtering
        all_detections = []
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = self.od_model.names[class_id]
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist() # [x1, y1, x2, y2]
            all_detections.append({'class_name': class_name, 'box': coords})

        print(f"Found {len(all_detections)} initial objects.")

        # --- Filtering Step: Remove Nested Bounding Boxes ---
        outer_detections = filter_nested_boxes(all_detections)
        print(f"Found {len(outer_detections)} outermost objects after filtering.")

        # --- Stage 2: Process Outermost Boxes ---
        final_items = []
        original_image = Image.open(image_path).convert("RGB")

        for detection in outer_detections:
            class_name = detection['class_name']
            box_coords = detection['box']
            
            # Check if the detected class is specific enough (e.g., "Apple", "Tomato")
            if class_name.lower() not in self.generic_classes_for_classification:
                print(f"- Found specific item: '{class_name}'. Adding directly.")
                final_items.append(class_name)
            else:
                # If it's a generic class, crop it and pass to the classifier
                print(f"- Found generic item: '{class_name}'. Cropping for classification...")
                x1, y1, x2, y2 = box_coords
                cropped_image = original_image.crop((x1, y1, x2, y2))
                
                # Run classification on the cropped region
                specific_item_name = self._classify_cropped_image(cropped_image)
                print(f"  -> Classified as: '{specific_item_name}'")
                final_items.append(specific_item_name)

        # --- Stage 3: Calorie Calculation ---
        print("\n--- Final Results and Calorie Calculation ---")
        calorie_summary = {}
        if not final_items:
            print("No food items were finalized for calorie calculation.")
        else:
            for item in final_items:
                calories = get_calories(item)
                calorie_summary[item] = calories
                print(f"Item: {item}, Calories: {calories}")
        
        return calorie_summary