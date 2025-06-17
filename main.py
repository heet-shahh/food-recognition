# food_pipeline/main.py

import os
from pipeline import FoodPipeline

def main():
    # --- Configuration ---
    # Define paths to your models and the test image
    OD_MODEL_PATH = "models/object_detection_best.pt"
    CLASSIFICATION_MODEL_PATH = "models/food_classifier_best.pt"
    TEST_IMAGE_PATH = "sample_image.jpg"

    # --- Pre-computation Check ---
    # Check if model files exist before initializing the pipeline
    if not os.path.exists(OD_MODEL_PATH):
        print(f"Error: Object detection model not found at {OD_MODEL_PATH}")
        print("Please place your trained OD model in the 'models' directory.")
        return

    if not os.path.exists(CLASSIFICATION_MODEL_PATH):
        print(f"Error: Classification model not found at {CLASSIFICATION_MODEL_PATH}")
        print("Please place your trained classification model in the 'models' directory.")
        return

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        print("Please add a 'sample_image.jpg' to the root directory.")
        return
        
    # --- Initialize and Run Pipeline ---
    try:
        # Create an instance of the pipeline
        food_pipeline = FoodPipeline(
            od_model_path=OD_MODEL_PATH,
            classification_model_path=CLASSIFICATION_MODEL_PATH
        )

        # Run inference on the test image
        final_results = food_pipeline.run_inference(TEST_IMAGE_PATH)

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


if __name__ == "__main__":
    main()

