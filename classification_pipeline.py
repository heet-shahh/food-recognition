import torch
import torch.nn as nn
import json
from PIL import Image
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from typing import Dict
from utils.calorie_database import get_nutrition_from_db

CLUBBED_ITEM_GROUPS = {
    "cheese": ["cheese", "tofu", "tempeh", "cottage_cheese"],
    "tofu": ["cheese", "tofu", "tempeh", "cottage_cheese"],
    "tempeh": ["cheese", "tofu", "tempeh", "cottage_cheese"],
    "cottage_cheese": ["cheese", "tofu", "tempeh", "cottage_cheese"],
    "greek_yogurt": ["greek_yogurt", "labneh"],
    "labneh": ["greek_yogurt", "labneh"],
    "bread": ["bread", "white_bread", "whole_wheat_bread"],
    "white_bread": ["bread", "white_bread", "whole_wheat_bread"],
    "whole_wheat_bread": ["bread", "white_bread", "whole_wheat_bread"],
}


class FoodPipeline:
    def __init__(self, cls_model_path: str, cls_mapping_path: str, num_classes: int):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on device: {self.device}")

        # Load classification model
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.cls_model = efficientnet_v2_s(weights=weights)
        num_ftrs = self.cls_model.classifier[1].in_features
        self.cls_model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.37, inplace=True),
            nn.Linear(num_ftrs, num_classes),
        )
        self.cls_model.load_state_dict(torch.load(cls_model_path, map_location=self.device))
        self.cls_model.to(self.device)
        self.cls_model.eval()

        # Transforms and class mapping
        self.cls_preprocess = weights.transforms()
        with open(cls_mapping_path, "r") as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def run_inference(self, image_path: str, cls_confidence_thresh: float = 0.35) -> Dict[str, Dict]:
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.cls_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.cls_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)\
        
        class_name = self.idx_to_class.get(pred_idx.item(), "Unknown")
        conf_score = confidence.item()
        
        if conf_score < cls_confidence_thresh:
            print(f"Low confidence ({conf_score:.2f}) for {class_name}. Skipping.")
            return {}

        # Load nutrition database
        with open("utils/local_nutrition_db.json") as f:
            nutrition_db = json.load(f)
        calorie_summary = {}

        # Handle clubbed items
        if class_name.lower() in CLUBBED_ITEM_GROUPS:
            print(f"Clubbing nutrients for: {class_name}")
            for alt_item in CLUBBED_ITEM_GROUPS[class_name.lower()]:
                if alt_item in nutrition_db:
                    calorie_summary[alt_item] = get_nutrition_from_db(alt_item, nutrition_db)
        else:
            if class_name.lower() in nutrition_db:
                print(f"Fetching nutrition for: {class_name}")
                calorie_summary[class_name] = get_nutrition_from_db(class_name, nutrition_db)

        return calorie_summary
