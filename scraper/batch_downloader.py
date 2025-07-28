import os
import json
import subprocess
import sys

# Paths
JSON_PATH = "dataset/item_with_image_ids.json"
DOWNLOADER_SCRIPT = "openimages_downloader.py"
BASE_OUTPUT_DIR = "dataset/images"
TMP_ID_LIST_DIR = "dataset/tmp_id_lists"
NUM_PROCESSES = "5"

os.makedirs(TMP_ID_LIST_DIR, exist_ok=True)

with open(JSON_PATH, "r") as f:
    food_image_map = json.load(f)

for food_name, image_ids in food_image_map.items():
    out_dir = os.path.join(BASE_OUTPUT_DIR, food_name)
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(TMP_ID_LIST_DIR, f"{food_name}_ids.txt")
    with open(txt_path, "w") as txt_f:
        for image_id in image_ids:
            txt_f.write(f"train/{image_id}\n")

    # Use the same Python interpreter that’s running this script:
    cmd = [
        sys.executable,           # ≤ this ensures boto3 is available
        DOWNLOADER_SCRIPT,
        txt_path,
        f"--download_folder={out_dir}",
        f"--num_processes={NUM_PROCESSES}"
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Download failed for {food_name}: exit {e.returncode}")

print("All batches submitted.")
