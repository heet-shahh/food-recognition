# food_pipeline/main.py
import os
from pipeline import FoodPipeline

def main():
    # --- Configuration ---
    OD_MODEL_PATH = "models/yolov8m-oiv7/train4/weights/best.pt"
    CLS_MODEL_PATH = "models/efficientnetv2s/best_model.pth"
    CLS_MAPPING_PATH = "class_mapping.json"
    NUM_CLASSES = 64
    IMAGE_ROOT_DIR = "dataset/images/test"
    OUTPUT_ROOT_DIR = "outputs/pipeline"

    # --- Pre-check ---
    required_files = [OD_MODEL_PATH, CLS_MODEL_PATH, CLS_MAPPING_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found at '{file_path}'")
            return

    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)

    # --- Initialize Pipeline ---
    try:
        food_pipeline = FoodPipeline(
            od_model_path=OD_MODEL_PATH,
            cls_model_path=CLS_MODEL_PATH,
            cls_mapping_path=CLS_MAPPING_PATH,
            num_classes=NUM_CLASSES
        )
    except Exception as e:
        print(f"Failed to initialize FoodPipeline: {e}")
        return

    # --- Traverse All Subdirectories ---
    for root, _, files in os.walk(IMAGE_ROOT_DIR):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue

            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(image_path, IMAGE_ROOT_DIR)
            output_path = os.path.join(OUTPUT_ROOT_DIR, relative_path)

            # Create subdirectory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                print(f"\nProcessing: {image_path}")
                final_results = food_pipeline.run_inference(
                    image_path=image_path,
                    od_confidence_thresh=0.3,
                    cls_confidence_thresh=0.35,
                    visualize=True,
                    filtering_method="advanced",
                    save_viz_path=output_path
                )

                print("  => Results:")
                if final_results:
                    for item, calories in final_results.items():
                        print(f"     - {item}: {calories}")
                else:
                    print("     - No items detected.")
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
