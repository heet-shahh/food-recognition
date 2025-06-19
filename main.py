# food_pipeline/main.py
import os
from pipeline import FoodPipeline

def main():
    # --- Configuration ---
    # Define paths to your models and the test image
    OD_MODEL_PATH = "models/yolov8m-oiv7/train4/weights/best.pt"
    CLS_MODEL_PATH = "models/efficientnetv2s/best_model.pth"
    TEST_IMAGE_PATH = "dataset/images/test/Balaleet/train_107.jpg"
    CLS_MAPPING_PATH = "class_mapping.json"
    NUM_CLASSES = 64

    # --- Pre-computation Check ---
    # Check if all required files exist before initializing the pipeline
    required_files = [OD_MODEL_PATH, CLS_MODEL_PATH, CLS_MAPPING_PATH, TEST_IMAGE_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found at '{file_path}'")
            return
        
    # --- Initialize and Run Pipeline ---
    try:
        food_pipeline = FoodPipeline(
            od_model_path=OD_MODEL_PATH,
            cls_model_path=CLS_MODEL_PATH,
            cls_mapping_path=CLS_MAPPING_PATH,
            num_classes=NUM_CLASSES
        )

        final_results = food_pipeline.run_inference(
            image_path=TEST_IMAGE_PATH,
            od_confidence_thresh=0.3,   # Optional: adjust confidence for OD model
            cls_confidence_thresh=0.35,   # Optional: adjust confidence for classifier
            visualize=True,
            filtering_method="advanced",
            save_viz_path="outputs/pipeline/Balaleet_train_107.jpg"
        )

        print("\n==================== SUMMARY ====================")
        print(f"Processed image: {TEST_IMAGE_PATH}")
        if final_results:
            for item, calories in final_results.items():
                print(f"- {item}: {calories}")
        else:
            print("No items detected or finalized.")
        print("==============================================")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred during the pipeline execution: {e}")
        # For debugging, you might want to see the full traceback
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
