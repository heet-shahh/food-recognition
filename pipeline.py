# food_pipeline/pipeline.py
import torch
import torch.nn as nn
import json
from ultralytics import YOLO
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Any
from utils.calorie_database import get_nutrition_from_db
from utils.enhanced_filtering import (
    filter_overlapping_boxes,
    filter_nested_boxes,
)
class FoodPipeline:
    def __init__(
        self,
        od_model_path: str,
        cls_model_path: str,
        cls_mapping_path: str,
        num_classes: int,
    ):
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
        with open(cls_mapping_path, "r") as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        # Define classes that trigger the second-stage classification
        self.generic_classes_for_classification = {
            "food",
            "drink",
            "plate",
            "bowl",
            "bottle",
            "dessert",
            "fast food",
            "coffee cup",
            "baked goods",
            "dairy product",
            "hamburger",
            "mixing bowl",
            "mug",
            "salad",
            "sandwich",
            "serving tray",
            "taco",
            "wine glass",
            "fruit",
            "Kitchen utensil","Kitchenware"}

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

    def _get_object_detection_results(
        self,
        image_path: str,
        od_confidence_thresh: float,
        filtering_method: str = "advanced",
    ) -> List[Dict[str, Any]]:
        """
        Get object detection results and apply filtering.

        Args:
            image_path: Path to image
            od_confidence_thresh: Confidence threshold for detections
            filtering_method: "basic", "advanced"
        """
        results = self.od_model.predict(
            image_path, conf=od_confidence_thresh, verbose=False
        )
        result = results[0]

        all_detections = []
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = self.od_model.names[class_id]
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            confidence = float(box.conf[0])  # Get confidence score
            all_detections.append(
                {"class_name": class_name, "box": coords, "confidence": confidence}
            )

        print(f"Found {len(all_detections)} initial objects.")

        # Apply different filtering methods
        if filtering_method == "basic":
            # Original method - only removes nested boxes
            filtered_detections = filter_nested_boxes(all_detections)
        elif filtering_method == "advanced":
            # Advanced method - handles overlapping boxes using IOU
            filtered_detections = filter_overlapping_boxes(
                all_detections,
                self.generic_classes_for_classification,
                iou_threshold=0.8,
            )
        else:
            raise ValueError(f"Unknown filtering method: {filtering_method}")

        print(f"Final filtered detections: {len(filtered_detections)}")
        return filtered_detections

    def _get_final_items(
        self, image_path: str, main_boxes, cls_confidence_thresh: float
    ):

        final_items = []
        original_image = Image.open(image_path).convert("RGB")
        classification_results = {}  # Store classification results for visualization

        for i, detection in enumerate(main_boxes):
            class_name = detection["class_name"]

            # If the OD detection is specific, use it directly
            if class_name.lower() not in self.generic_classes_for_classification:
                print(f"- Found OD item: '{class_name}'. Adding directly.")
                final_items.append(class_name)
            else:
                # Otherwise, crop the box and pass it to the classifier
                print(
                    f"- Found matching OD item: '{class_name}'. Cropping for classification..."
                )
                x1, y1, x2, y2 = detection["box"]
                cropped_image = original_image.crop((x1, y1, x2, y2))

                specific_item_name, confidence = self._classify_cropped_image(
                    cropped_image
                )
                classification_results[i] = (
                    specific_item_name,
                    confidence,
                )  # Store for visualization

                # Only accept the classification if confidence is high enough
                if confidence >= cls_confidence_thresh:
                    print(
                        f"  -> Classified as '{specific_item_name}' with confidence {confidence:.2f}. ACCEPTED."
                    )
                    final_items.append(specific_item_name)
                else:
                    # If confidence is low, fall back to the generic OD label
                    print(
                        f"  -> Classified as '{specific_item_name}' with confidence {confidence:.2f}. REJECTED."
                    )

        return final_items, classification_results

    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        classification_results: Dict = None,
        save_path: str = None,
        title: str = "Detection Results",
    ):
        """
        Visualize bounding boxes and class labels on the image.
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Create figure and axis
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.set_title(title, fontsize=16, fontweight="bold")

            # Define a set of colors
            colors = [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ]

            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection["box"]
                class_name = detection["class_name"]
                color = colors[i % len(colors)]

                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add label
                label_text = f"{class_name}"
                if classification_results and i in classification_results:
                    cls_name, cls_conf = classification_results[i]
                    label_text += f"\n→ {cls_name} ({cls_conf:.2f})"

                ax.text(
                    x1,
                    y1 - 5,
                    label_text,
                    fontsize=10,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment="top",
                    weight="bold",
                )

            ax.axis("off")
            plt.tight_layout()

            if save_path:
                fig.savefig(
                    save_path, dpi=150, bbox_inches="tight"
                )  # Lower DPI to save memory
                print(f"Visualization saved to: {save_path}")

            plt.close(fig)  # ✅ Important to free memory

        except Exception as e:
            print(f"Error visualizing detections: {e}")

    def run_inference(
        self,
        image_path: str,
        od_confidence_thresh=0.25,
        cls_confidence_thresh=0.5,
        filtering_method="advanced",
        visualize=False,
        save_viz_path=None,
    ) -> Dict[str, str]:
        """
        Executes the full two-stage pipeline on a single image.

        Args:
            image_path: Path to the input image
            od_confidence_thresh: Confidence threshold for object detection
            cls_confidence_thresh: Confidence threshold for classification
            filtering_method: "basic", "advanced"
            visualize: Whether to show visualization
            save_viz_path: Path to save visualization (optional)
        """
        print(f"\n--- Starting inference for {image_path} ---")
        print(f"Using filtering method: {filtering_method}")

        # --- Stage 1: Object Detection with Enhanced Filtering ---
        main_boxes = self._get_object_detection_results(
            image_path, od_confidence_thresh, filtering_method
        )

        final_items = []
        classification_results = {}

        # --- Stage 2: Process Detections or Classify Whole Image ---
        if not main_boxes:
            # CHANGE: If no boxes are detected, classify the entire image
            print(
                "\nNo bounding boxes detected. Passing the entire image to the classification model."
            )
            original_image = Image.open(image_path).convert("RGB")

            # Classify the entire image
            specific_item_name, confidence = self._classify_cropped_image(
                original_image
            )

            # Only accept the classification if confidence is high enough
            if confidence >= cls_confidence_thresh:
                print(
                    f"  -> Whole image classified as '{specific_item_name}' with confidence {confidence:.2f}. ACCEPTED."
                )
                final_items.append(specific_item_name)
            else:
                print(
                    f"  -> Whole image classified as '{specific_item_name}' with confidence {confidence:.2f}. REJECTED."
                )
        else:
            # If boxes are found, proceed with the original logic
            final_items, classification_results = self._get_final_items(
                image_path, main_boxes, cls_confidence_thresh
            )

        # Visualize final results with classification if requested
        if visualize:
            self.visualize_detections(
                image_path,
                main_boxes,  # Will be empty if no detections were found
                classification_results,
                title="Final Results with Classification",
                save_path=save_viz_path,
            )

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
                
            with open("utils/local_nutrition_db.json") as f:
                nutrition_db = json.load(f)

            for item, count in item_counts.items():
                nutrients = get_nutrition_from_db(item, nutrition_db)
                # Add count to the key if more than one instance is found
                display_key = f"{item} (x{count})" if count > 1 else item
                calorie_summary[display_key] = nutrients
                print(f"Item: {display_key}, nutrients: {nutrients}")

        return calorie_summary
