import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoImageProcessor
from PIL import Image
import os
import numpy as np
import albumentations as A


class DETRDataLoader:
    def __init__(self, dataset_format, image_height=480, image_width=1920, load_model_hub_id="facebook", 
                 load_model_repo_id="detr-resnet-50", dataset_hub_id="ARG-NCTU", 
                 dataset_repo_id="Kaohsiung_Port_dataset_2024", classes_path="data/classes.txt"):
        """
        Initializes the DETR dataset class.

        Args:
            dataset_format (str): Dataset format, either 'jsonl' or 'parquet'.
            image_height (int): Height of the input image.
            image_width (int): Width of the input image.
            load_model_hub_id (str): Hugging Face hub ID for loading model.
            load_model_repo_id (str): Hugging Face repo ID for loading model.
            dataset_hub_id (str): Hugging Face hub ID for dataset.
            dataset_repo_id (str): Dataset repository ID.
            classes_path (str): Path to class labels file.
        """
        self.dataset_format = dataset_format
        self.image_height = image_height
        self.image_width = image_width
        self.load_model_hub_id = load_model_hub_id
        self.load_model_repo_id = load_model_repo_id
        self.dataset_hub_id = dataset_hub_id
        self.dataset_repo_id = dataset_repo_id
        self.classes_path = classes_path

        # Load class mappings
        self.id2label = self.get_id2label()
        self.label2id = self.get_label2id()

        # Initialize image processor and transforms
        self.image_processor = self.get_image_processor()
        self.transform = self.get_transform()
        self.collate_fn = self.get_collate_fn()

        # Load dataset
        self.dataset = self.get_dataset()

    def load_classes(self):
        """Load class labels from file."""
        with open(self.classes_path, "r") as f:
            return [cname.strip() for cname in f.readlines()]

    def get_id2label(self):
        """Create an ID-to-label mapping."""
        return {index: x for index, x in enumerate(self.load_classes(), start=0)}

    def get_label2id(self):
        """Create a label-to-ID mapping."""
        return {v: k for k, v in self.get_id2label().items()}

    def get_image_processor(self):
        """Return the image processor for DETR."""
        processor = AutoImageProcessor.from_pretrained(f"{self.load_model_hub_id}/{self.load_model_repo_id}")
        processor.size = {"height": self.image_height, "width": self.image_width}
        return processor

    def get_transform(self):
        """Return the image augmentation pipeline."""
        return A.Compose(
            [
                # A.Resize(800, 800),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
        )

    def get_collate_fn(self):
        def collate_fn(batch):
            pixel_values = [item["pixel_values"] for item in batch]
            encoding = self.image_processor.pad(pixel_values, return_tensors="pt")
            labels = [item["labels"] for item in batch]
            batch = {}
            batch["pixel_values"] = encoding["pixel_values"]
            batch["pixel_mask"] = encoding["pixel_mask"]
            batch["labels"] = labels
            return batch
        return collate_fn

    def formatted_anns(self, image_id, category, area, bbox):
        """Format annotations for training."""
        return [
            {"image_id": image_id, "category_id": category[i], "isCrowd": 0, "area": area[i], "bbox": list(bbox[i])}
            for i in range(len(category))
        ]

    def transform_aug_ann(self, examples):
        """Apply transformations to a batch of images and annotations."""
        image_ids = examples["image_id"]
        images, bboxes, areas, categories = [], [], [], []

        for image_path, objects in zip(examples["image_path"], examples["objects"]):
            image = self.load_image(image_path)

            try:
                # Apply transformations
                transformed = self.transform(image=image, bboxes=objects["bbox"], category=objects["category"])
                image, bboxes_trans, categories_trans = transformed["image"], transformed["bboxes"], transformed["category"]
            except Exception as e:
                print(f"Transform error: {e}. Using default bbox.")
                image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)  # Black placeholder image
                bboxes_trans, categories_trans = [[0.4, 0.4, 0.2, 0.2]], [0]  # Default bbox & category

            images.append(image)
            bboxes.append(bboxes_trans)
            areas.append(objects["area"])
            categories.append(categories_trans)

        # Format annotations correctly
        targets = [
            {"image_id": id_, "annotations": self.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, areas, bboxes)
        ]

        return self.image_processor(images=images, annotations=targets, return_tensors="pt")

    def load_image(self, image_path):
        """Load an image from JSONL or Parquet dataset."""
        try:
            return np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]
        except Exception:
            print(f"Warning: {image_path} not found, using black placeholder.")
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

    def get_dataset(self):
        """Load dataset and apply transformations using `with_transform()`."""
        if self.dataset_format == "jsonl":
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
        else:
            dataset = load_dataset(f"{self.dataset_hub_id}/{self.dataset_repo_id}")

        dataset["train"] = dataset["train"].with_transform(lambda x: self.transform_aug_ann(x))
        return dataset

    def get_dataloader(self, batch_size=8):
        """Return a DataLoader for training."""
        return DataLoader(self.dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
