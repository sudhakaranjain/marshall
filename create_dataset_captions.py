import argparse
import json
import os

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing MS COCO dataset - Specific class", add_help=True)

    parser.add_argument("--dataset-path", type=str, default="./datasets", help="Path to the dataset")
    parser.add_argument("--dataset-split", type=str, default="train", help="Category of split i.e. 'train' or 'val'")
    parser.add_argument("--annotations-path", type=str, default="annotations_trainval2017/annotations/",
                        help="Path to the MS COCO annotations")

    args = parser.parse_args()

    # Load the captions
    with open(os.path.join(args.dataset_path, args.annotations_path, f"captions_{args.dataset_split}2017.json")) as f:
        captions = json.load(f)

    all_instances = []
    for image in tqdm(captions['images']):
        annotation_list = captions['annotations']
        image_captions = [annotation['caption'] for annotation in annotation_list
                          if annotation['image_id'] == image['id']]
        all_instances.append({"file_name": image["file_name"], "captions": image_captions})

    # # Create file to store dataset information
    captions_file_path = os.path.join(args.dataset_path, f"{args.dataset_split}_captions.json")
    with open(captions_file_path, "w") as f:
        json.dump(all_instances, f)
