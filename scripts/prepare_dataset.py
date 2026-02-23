"""
Prepare the Alzheimer's dataset from raw images + label file.

Reads data_all_files_2026.txt for MOCA scores, copies images to
alzhimer_src_images/<collection>/, and generates point clouds
saved to alzhimer_point_clouds/<collection>/<stem>.pt.

Each .pt file contains: {'point_cloud': Tensor(1024, 2), 'label': int}
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from PIL import Image
from util import CustomSketchToPointCloud

ROOT = os.path.join(os.path.dirname(__file__), '..')

LABEL_FILE = os.path.join(ROOT, 'data_all_files_2026.txt')
SRC_DIR = os.path.join(ROOT, 'alzheimer_2026_all_blended_512')
IMG_OUT = os.path.join(ROOT, 'alzhimer_src_images')
PC_OUT = os.path.join(ROOT, 'alzhimer_point_clouds')

NUM_POINTS = 1024
GRID_SIZE = 4


def main():
    # Parse label file
    labels = {}
    with open(LABEL_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            score = float(parts[0])
            path = parts[1]
            labels[path] = score

    print(f'Loaded {len(labels)} labels from {LABEL_FILE}')

    converter = CustomSketchToPointCloud(num_points=NUM_POINTS, grid_size=GRID_SIZE)

    success = 0
    skipped = 0
    failed = 0
    failed_files = []

    for rel_path, moca_score in labels.items():
        collection, filename = rel_path.split('/', 1)
        stem = os.path.splitext(filename)[0]

        src_img = os.path.join(SRC_DIR, collection, filename)
        if not os.path.exists(src_img):
            print(f'MISSING: {src_img}')
            skipped += 1
            continue

        os.makedirs(os.path.join(IMG_OUT, collection), exist_ok=True)
        os.makedirs(os.path.join(PC_OUT, collection), exist_ok=True)

        # Copy image
        dst_img = os.path.join(IMG_OUT, collection, filename)
        shutil.copy2(src_img, dst_img)

        # Generate point cloud
        try:
            img = Image.open(src_img).convert('L')
            point_cloud = converter(img)       # (1024, 2)
            torch.save(
                {'point_cloud': point_cloud, 'label': int(moca_score)},
                os.path.join(PC_OUT, collection, f'{stem}.pt'),
            )
            success += 1
        except Exception as e:
            print(f'FAILED: {rel_path} - {e}')
            failed += 1
            failed_files.append(rel_path)

        if (success + failed) % 500 == 0 and (success + failed) > 0:
            print(f'Progress: {success + failed + skipped}/{len(labels)}...')

    print(f'\nDone: {success} success, {skipped} skipped (missing), {failed} failed')
    if failed_files:
        print('Failed files:')
        for f in failed_files:
            print(f'  {f}')


if __name__ == '__main__':
    main()
