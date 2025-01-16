from datasets import Features, Value, Sequence
from datasets import load_dataset
from transformers import AutoImageProcessor
import numpy as np
import os
from PIL import Image, ImageDraw
import albumentations
import numpy as np
import torch

from transformers import AutoModelForObjectDetection
from transformers import TrainingArguments
from transformers import Trainer

import json

from transformers import pipeline
import requests

import torchvision

from transformers import AutoModelForObjectDetection, AutoImageProcessor
import evaluate
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt
from transformers import pipeline, AutoImageProcessor, AutoModelForObjectDetection
from torchvision.ops import box_iou

############# Data Loading #############

# Specify the correct schema for your dataset
features = Features({
    'image_id': Value('int32'),
    'image_path': Value('string'),
    'width': Value('int32'),
    'height': Value('int32'),
    'objects': {
        'id': Sequence(Value('int32')),
        'area': Sequence(Value('float32')), 
        'bbox': Sequence(Sequence(Value('float32'), length=4)), 
        'category': Sequence(Value('int32'))
    }
})


dataset = load_dataset(
    'json', 
    data_files={'train': 'data/instances_train2024_rvrr.jsonl', 
                'validation': 'data/instances_val2024_rvrr.jsonl'},
    features=features
)

print('\n\n')
print(dataset["train"][0])
print('\n\n')

def load_classes():
    class_list = []
    with open("data/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()



image = Image.open(dataset["train"][0]["image_path"])
annotations = dataset["train"][0]["objects"]
draw = ImageDraw.Draw(image)

# categories = dataset["train"].features["objects"].feature["category"].names

id2label = {index: x for index, x in enumerate(class_list, start=0)}
label2id = {v: k for k, v in id2label.items()}

for i in range(len(annotations["id"])):
    box = annotations["bbox"][i - 1]
    class_idx = annotations["category"][i - 1]
    x, y, w, h = tuple(box)
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")

# save the image
image.save("visualize_anno.jpg")

############# Preprocessing #############

checkpoint = "ARG-NCTU/detr-resnet-50-finetuned-20-epochs-boat-dataset"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        # albumentations.HorizontalFlip(p=1.0),
        albumentations.HorizontalFlip(p=0.5),
        # albumentations.RandomBrightnessContrast(p=1.0),
        albumentations.RandomBrightnessContrast(p=0.5),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

# Create an empty placeholder image
def create_empty_image(width=640, height=480, color=(0, 0, 0)):
    return np.zeros((height, width, 3), dtype=np.uint8)  # Black image by default


# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image_path, objects in zip(examples["image_path"], examples["objects"]):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'{image_path} does not exist, using a placeholder image.')

            # Try opening the image
            image = Image.open(image_path)
            image = np.array(image.convert("RGB"))[:, :, ::-1]
        
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(e)
            # Use a black placeholder image if the actual image is missing or cannot be opened
            image = create_empty_image()
        
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")

dataset["train"] = dataset["train"].with_transform(transform_aug_ann)
print('\n\n')
print(dataset["train"][15])
print('\n\n')

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    # encoding = image_processor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

############# Evaluation #############

# format annotations the same as for training, no need for data augmentation
def val_formatted_anns(image_id, objects):
    annotations = []
    for i in range(0, len(objects["id"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i],
        }
        annotations.append(new_ann)

    return annotations


# Save images and annotations into the files torchvision.datasets.CocoDetection expects
def save_annotation_file_images(dataset, mode="val"):
    output_json = {}
    path_output = f"{os.getcwd()}/output/"

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    if mode == "val":
        path_anno = os.path.join(path_output, "boat_ann_val.json")
    else:
        path_anno = os.path.join(path_output, "boat_ann_val_real.json")
    categories_json = [{"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label]
    output_json["images"] = []
    output_json["annotations"] = []
    for example in dataset:
        ann = val_formatted_anns(example["image_id"], example["objects"])
        if not os.path.exists(example["image_path"]):
            continue
        image_example = Image.open(example["image_path"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": image_example.width,
                "height": image_example.height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    for image_path, img_id in zip(dataset["image_path"], dataset["image_id"]):
        if not os.path.exists(image_path):
            continue
        im = Image.open(image_path)
        path_img = os.path.join(path_output, f"{img_id}.png")
        im.save(path_img)

    return path_output, path_anno

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}

def eval(eval_dataset, mode="val"):
    im_processor = AutoImageProcessor.from_pretrained("ARG-NCTU/detr-resnet-50-finetuned-30-epochs-boat-dataset")
    path_output, path_anno = save_annotation_file_images(eval_dataset, mode)
    test_ds_coco_format = CocoDetection(path_output, im_processor, path_anno)
    
    model = AutoModelForObjectDetection.from_pretrained("ARG-NCTU/detr-resnet-50-finetuned-30-epochs-boat-dataset")
    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]
    
            labels = [
                {k: v for k, v in t.items()} for t in batch["labels"]
            ]  # these are in DETR format, resized + normalized
    
            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = im_processor.post_process(outputs, orig_target_sizes)  # convert outputs of model to COCO api
    
            module.add(prediction=results, reference=labels)
            del batch
    
    results = module.compute()
    print(results)

eval(dataset["validation"], mode="val")



