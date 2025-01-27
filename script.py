from utils.ControlNet import generate_canny_image, load_models, create_pipeline, generate_image, generate_canny_from_folder,generate_depth_map
import torch
import gc
import os
import math
import numpy as np
import torch
import safetensors.torch as sf

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
import random
import cv2



import warnings
warnings.filterwarnings("ignore")



def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

flush()

# canny_image = generate_canny_from_folder("trash/to_send/image1.jpeg")
canny_image = generate_canny_image(image_url="https://i.guim.co.uk/img/media/862408555e528f7f91e93f7912f0ac52e372b7b3/0_234_3242_1946/master/3242.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=348ea4266ad81c0daf1dacfff99502bb")
# depth_map = generate_depth_map(image_url="https://i.guim.co.uk/img/media/862408555e528f7f91e93f7912f0ac52e372b7b3/0_234_3242_1946/master/3242.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=348ea4266ad81c0daf1dacfff99502bb")


controlnet, transformer = load_models(controlnet_model="InstantX/FLUX.1-dev-controlnet-canny", transformer_checkpoint="https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_0.gguf")
# controlnet, transformer = load_models(controlnet_model="Shakker-Labs/FLUX.1-dev-ControlNet-Depth", transformer_checkpoint="https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_0.gguf")

pipe = create_pipeline(controlnet, transformer)

with torch.no_grad():
    output_image = generate_image(pipe, "high quality, realistic image of a british man of age 21, standing outside a church, happily posing in the afternoon", canny_image, canny_image.width, canny_image.height)
# output_image = generate_image(pipe, "high quality, realistic image of a british man of age 21, standing outside a church, happily posing in the afternoon", depth_map, depth_map.width, depth_map.height)

flush()





