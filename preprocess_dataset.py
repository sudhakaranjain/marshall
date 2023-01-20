import argparse
import json
import os

from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing MS COCO dataset - Specific class", add_help=True)

    parser.add_argument("--dataset-path", type=str, default="./datasets", help="Path to the dataset")
    parser.add_argument("--dataset-split", type=str, default="train", help="Category of split i.e. 'train' or 'val'")
    parser.add_argument("--annotations-path", type=str, default="annotations_trainval2017/annotations/",
                        help="Path to the MS COCO annotations")
    parser.add_argument("--extract-class-id", type=int, default=3, help="Class id of the class one wants to extract")
    parser.add_argument("--limit", type=int, default=0,
                        help="If non-zero positive number, then limit the number of extracted instances to the "
                             "corresponding number else don't limit")

    args = parser.parse_args()

    dataset_split = args.dataset_split

    # Load the annotations
    with open(os.path.join(args.dataset_path, args.annotations_path, f"instances_{dataset_split}2017.json")) as f:
        annotations = json.load(f)

    # Load their corresponding captions
    with open(os.path.join(args.dataset_path, args.annotations_path, f"captions_{dataset_split}2017.json")) as f:
        captions = json.load(f)

    # Create directory for the extracted dataset
    extract_class_id = args.extract_class_id
    extract_data_path = os.path.join(args.dataset_path, f"class_{extract_class_id}_{dataset_split}")
    os.makedirs(extract_data_path, exist_ok=True)

    category_image_ids = set()
    images_data_path = os.path.join(args.dataset_path, f"{dataset_split}2017")
    positive_extracted_count, negative_extracted_count = 0, 0
    negative_example = 0
    # For every positive class also collect a negative class
    all_instances = []
    for ann in tqdm(annotations["annotations"]):
        if ann["category_id"] == extract_class_id and ann["image_id"] not in category_image_ids:
            category_image_ids.add(ann["image_id"])
            negative_example += 1
            label = 1
        elif negative_example and ann["image_id"] not in category_image_ids:
            category_image_ids.add(ann["image_id"])
            negative_example -= 1
            label = 0
        else:
            continue

        for image in annotations["images"]:
            if image["id"] == ann["image_id"]:
                im = Image.open(os.path.join(images_data_path, image["file_name"]))
                if len(im.split()) < 3:
                	break
                im.save(os.path.join(extract_data_path, image["file_name"]))
                if label:
                    positive_extracted_count += 1
                else:
                    negative_extracted_count += 1

                for caption in captions["annotations"]:
                    if caption["image_id"] == image["id"]:
                        instance = {"file_name": image["file_name"], "caption": caption["caption"], "label": label}
                        all_instances.append(instance)
                        break
                break

        if 0 < args.limit < positive_extracted_count:
            break

    with open(os.path.join(extract_data_path, f"class_{extract_class_id}_{dataset_split}.json"), "w") as f:
        json.dump(all_instances, f)

    print(f"{positive_extracted_count} instances of class {extract_class_id} and {negative_extracted_count} of "
          f"negative instances in {dataset_split}2017 dataset were extracted!")
