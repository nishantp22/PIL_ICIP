import torch
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from diffusers import FluxControlNetPipeline
from diffusers.models import FluxControlNetModel
from diffusers.utils import load_image
from PIL import Image
import os
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests


def crop_to_16(image):
    width, height = image.size
    
    new_width = (width // 16) * 16
    new_height = (height // 16) * 16

    left = 0
    upper = 0
    right = new_width
    lower = new_height

    cropped_image = image.crop((left, upper, right, lower))
    
    return cropped_image


def load_image_from_folder(image_path):
    """
    Load an image from a local folder.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image: The loaded image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    return Image.open(image_path)


def generate_canny_from_folder(image_path, save_path="images/canny_image.png"):
    """
    Generate a Canny edge-detected image from a local folder image and save it.

    Args:
        image_path (str): Path to the original image.
        save_path (str): Path to save the Canny image.

    Returns:
        PIL.Image: The Canny edge-detected image.
    """
    original_image = load_image_from_folder(image_path)
    cropped_image = crop_to_16(original_image)

    cropped_image.save("images/original_image.png")
    image = np.array(cropped_image.convert("L"))  # Convert to grayscale

    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Convert single channel edges to 3-channel image
    edges_colored = np.stack([edges]*3, axis=-1)

    canny_image = Image.fromarray(edges_colored)
    canny_image.save(save_path)
    return canny_image

def generate_canny_image(image_url, save_path="images/canny_image.png"):
    """
    Generate a Canny edge-detected image from a given URL and save it.

    Args:
        image_url (str): URL of the original image.
        save_path (str): Path to save the Canny image.

    Returns:
        PIL.Image: The Canny edge-detected image.
    """
    original_image = load_image(image_url)
    cropped_image=crop_to_16(original_image)


    # image = Image.fromarray(cropped_image)
    cropped_image.save("images/original_image.png")
    image = np.array(cropped_image)

    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)


    canny_image = Image.fromarray(image)
    canny_image.save(save_path)
    return canny_image


def generate_depth_map(image_url, save_path="images/depth_map.png"):

    original_image = Image.open(requests.get(image_url, stream=True).raw)
    image = crop_to_16(original_image)
    image.save("images/original_image.png")

    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8")).convert("RGB")

    depth.save(save_path)


    return depth


def load_models(controlnet_model, transformer_checkpoint):
    """
    Load the ControlNet and transformer models.

    Args:
        controlnet_model (str): Model identifier for the ControlNet.
        transformer_checkpoint (str): URL or path to the transformer checkpoint.

    Returns:
        tuple: Loaded ControlNet model and transformer model.
    """
    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)

    transformer = FluxTransformer2DModel.from_single_file(
        transformer_checkpoint,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: "16GB", 1: "16GB"}
    )

    return controlnet, transformer

def create_pipeline(controlnet, transformer, pipeline_model="black-forest-labs/FLUX.1-dev"):
    """
    Create a pipeline using the provided ControlNet and transformer models.

    Args:
        controlnet: Pretrained ControlNet model.
        transformer: Pretrained transformer model.
        pipeline_model (str): Model identifier for the pipeline.

    Returns:
        FluxControlNetPipeline: Configured pipeline.
    """
    pipe = FluxControlNetPipeline.from_pretrained(
        pipeline_model,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        # device_map="balanced",
    )
    pipe.to("cuda")
    return pipe

def generate_image(pipe, prompt, control_image, width, height, save_path="images/controlnet_output.png"):
    """
    Generate an image using the pipeline and save it.

    Args:
        pipe (FluxControlNetPipeline): The pipeline to use.
        prompt (str): Text prompt for image generation.
        control_image (PIL.Image): The control image (e.g., Canny image).
        width (int): Width of the output image.
        height (int): Height of the output image.
        save_path (str): Path to save the generated image.

    Returns:
        PIL.Image: The generated image.
    """
    image = pipe(
        prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.7,
        num_inference_steps=40,
        guidance_scale=3.5,
        width=width,
        height=height,
    ).images[0]
    image.save(save_path)
    return image

