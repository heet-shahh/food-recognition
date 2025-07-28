# python3
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Open Images image downloader.

This script downloads a subset of Open Images images, given a list of image ids.
Typical uses of this tool might be downloading images:
- That contain a certain category.
- That have been annotated with certain types of annotations (e.g. Localized
Narratives, Exhaustively annotated people, etc.)

The input file IMAGE_LIST should be a text file containing one image per line
with the format <SPLIT>/<IMAGE_ID>, where <SPLIT> is either "train", "test",
"validation", or "challenge2018"; and <IMAGE_ID> is the image ID that uniquely
identifies the image in Open Images. A sample file could be:
  train/f9e0434389a1d4dd
  train/1a007563ebc18664
  test/ea8bfd4e765304db

"""

import argparse
from concurrent import futures
import os
import re
import sys
import csv

import boto3
import botocore
import tqdm

BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]+)'


def read_image_list_file(image_list_file):
    with open(image_list_file, 'r') as f:
        for line in f:
            yield line.strip().replace('.jpg', '')


def check_and_homogenize_image_list(image_list):
    for line_number, image in enumerate(image_list):
        m = re.match(REGEX, image)
        if not m:
            raise ValueError(
                f'ERROR in line {line_number} of the image list. '
                f'Not recognized: "{image}".'
            )
        yield m.group(1), m.group(2)


def download_one_image(bucket, split, image_id, download_folder):
    out_path = os.path.join(download_folder, f'{image_id}.jpg')
    try:
        bucket.download_file(f'{split}/{image_id}.jpg', out_path)
        return image_id, "Downloaded", ""
    except botocore.exceptions.ClientError as e:
        err = str(e)
        # Print to terminal immediately:
        print(f"❌ Failed: {split}/{image_id} — {err}")
        return image_id, "Failed", err
    except Exception as e:
        err = str(e)
        print(f"❌ Failed: {split}/{image_id} — {err}")
        return image_id, "Failed", err


def download_all_images(args):
    bucket = boto3.resource(
        's3', config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    ).Bucket(BUCKET_NAME)

    download_folder = args['download_folder'] or os.getcwd()
    os.makedirs(download_folder, exist_ok=True)

    # Read & validate
    raw_list = list(read_image_list_file(args['image_list']))
    try:
        image_list = list(check_and_homogenize_image_list(raw_list))
    except ValueError as e:
        sys.exit(e)

    # Prepare CSV log
    log_path = os.path.join(download_folder, 'download_log.csv')
    log_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Split", "ImageID", "Status", "Error"])

    # Download in parallel
    progress = tqdm.tqdm(total=len(image_list), desc='Downloading images', leave=True)
    with futures.ThreadPoolExecutor(max_workers=args['num_processes']) as executor:
        future_to_item = {
            executor.submit(download_one_image, bucket, split, img_id, download_folder): (split, img_id)
            for split, img_id in image_list
        }

        for future in futures.as_completed(future_to_item):
            split, _ = future_to_item[future]
            image_id, status, err = future.result()
            # Write to CSV
            csv_writer.writerow([split, image_id, status, err])
            progress.update(1)

    progress.close()
    log_file.close()

    print(f"\n✅ Done. Full log at: {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'image_list',
        type=str,
        help='Text file: one <split>/<image_id> per line (no .jpg)'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=5,
        help='Number of parallel threads (default: 5)'
    )
    parser.add_argument(
        '--download_folder',
        type=str,
        default=None,
        help='Where to put downloaded .jpg files (default: cwd)'
    )
    download_all_images(vars(parser.parse_args()))