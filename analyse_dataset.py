import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysing MS COCO dataset", add_help=True)

    parser.add_argument("--dataset-split", type=str, default="train", help="Category of split i.e. 'train' or 'val'")
    parser.add_argument("--annotations-path", type=str, default="./datasets/annotations_trainval2017/annotations/",
                        help="Path to the MS COCO annotations")

    args = parser.parse_args()

    # Load the annotations
    with open(os.path.join(args.annotations_path, f"instances_{args.dataset_split}2017.json")) as f:
        annotations = json.load(f)

    category_counts = dict()
    stats = []

    # Collect the categories
    for categories in tqdm(annotations["categories"]):
        category_counts[categories["id"]] = 0
        stats.append((categories["id"], categories["name"]))

    classes = [name for _, name in stats]
    print(f"All the categories in the dataset(n={len(classes)}): {classes}")

    category_image_ids = defaultdict(set)
    annotations_image_ids = set([images["id"] for images in annotations["images"]])

    for ann in tqdm(annotations["annotations"]):
        if ann["image_id"] not in category_image_ids[ann["category_id"]]:
            category_image_ids[ann["category_id"]].add(ann["image_id"])
            if ann["image_id"] in annotations_image_ids:
                category_counts[ann["category_id"]] += 1

    print("Annotation Stats:")
    stats = [(id, name, category_counts[id]) for id, name in stats]
    stats = sorted(stats, key=lambda x: x[-1], reverse=True)
    print(stats)
