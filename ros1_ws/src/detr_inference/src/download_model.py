#!/usr/bin/env python3
import os
from huggingface_hub import hf_hub_download
import rospkg
import rospy

rospack = rospkg.RosPack()

# Define the local path for the model
local_model_dir = os.path.join(rospack.get_path("detr_inference"), "model")        

# Load the processor and model
hf_account_name = "ARG-NCTU"
hf_repo_name = "detr-resnet-50-finetuned-20-epochs-boat-dataset"
hf_model_path = os.path.join(local_model_dir, hf_account_name, hf_repo_name)

if not os.path.exists(hf_model_path):
    # Download from Hugging Face and save locally
    if hf_account_name is None or hf_repo_name is None:
        raise ValueError("Hugging Face account name and repository name must be provided.")
    os.makedirs(hf_model_path, exist_ok=True)
    download_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    for file in download_files:
        hf_hub_download(repo_id=f"{hf_account_name}/{hf_repo_name}", repo_type="model", filename=file, local_dir=hf_model_path)
