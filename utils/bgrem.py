from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from transparent_background import Remover
from tqdm import tqdm

def tensor2pil(image):
    """
    Convert a PyTorch tensor to a PIL image.

    Args:
        image (torch.Tensor): The input tensor.

    Returns:
        PIL.Image: The converted PIL image.
    """
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """
    Convert a PIL image to a PyTorch tensor.

    Args:
        image (PIL.Image): The input PIL image.

    Returns:
        torch.Tensor: The converted tensor.
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class InspyrenetRembg:
    """
    Class for removing backgrounds from images.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input types for the class.

        Returns:
            dict: Required input types.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit):
        """
        Remove background from images.

        Args:
            image (torch.Tensor): Batch of images as tensors.
            torchscript_jit (str): Whether to use JIT optimization ("default" or "on").

        Returns:
            tuple: Processed images and their masks.
        """
        remover = Remover(jit=(torchscript_jit == "on"))
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type='rgba')
            out = pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        return (img_stack, mask)



class InspyrenetRembgAdvanced:
    """
    Advanced class for removing backgrounds with additional options.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input types for the class.

        Returns:
            dict: Required input types.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def dilate_mask(self, mask, kernel_size=25):
        """
        Apply dilation to extend the boundaries of the mask.

        Args:
            mask (torch.Tensor): The mask tensor (1 or 0 values).
            kernel_size (int): The size of the dilation kernel.

        Returns:
            torch.Tensor: The dilated mask.
        """
        
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
        dilated_mask = F.conv2d(mask.unsqueeze(1).float(), kernel, padding=kernel_size//2)
        dilated_mask = (dilated_mask > 0).float()
        return dilated_mask.squeeze(1)  


    def remove_background(self, image, torchscript_jit, threshold, dilate=True, dilation_kernel_size=25):
        """
        Remove background from images with threshold adjustment and optional dilation.

        Args:
            image (torch.Tensor): Batch of images as tensors.
            torchscript_jit (str): Whether to use JIT optimization ("default" or "on").
            threshold (float): Threshold for background removal.
            dilate (bool): Whether to apply dilation to the mask.
            dilation_kernel_size (int): Kernel size for dilation.

        Returns:
            tuple: Processed images and their masks.
        """
        remover = Remover(jit=(torchscript_jit == "on"))
        img_list = []
        for img in tqdm(image, "Inspyrenet Rembg"):
            mid = remover.process(tensor2pil(img), type='rgba', threshold=threshold)
            out = pil2tensor(mid)
            img_list.append(out)
        img_stack = torch.cat(img_list, dim=0)
        
        mask = img_stack[:, :, :, 3]
        
        if dilate:
            mask = self.dilate_mask(mask, kernel_size=dilation_kernel_size)
        
        return (img_stack, mask)



def apply_remove_background(image_path,  outName, save_dir="images", threshold=0.5,):
    """
    Process an image to remove its background and save results.

    Args:
        image_path (str): Path to the input image.
        save_dir (str): Directory to save the output images.
        threshold (float): Threshold for background removal.

    Returns:
        None
    """
    model = InspyrenetRembgAdvanced()

    pil_image = Image.open(image_path).convert("RGB")
    tensor_image = pil2tensor(pil_image)

    batch_of_images = tensor_image.repeat(1, 1, 1, 1)

    img_stack, mask = model.remove_background(batch_of_images, "default", threshold=threshold,dilate=True)

    output = tensor2pil(img_stack[0])
    output.save(f"{save_dir}/{outName}.png")

    mask = tensor2pil(mask.unsqueeze(0))
    mask.save(f"{save_dir}/{outName}Mask.png")

    print("Output image and Mask saved.")