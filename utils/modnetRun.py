import os
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .modnet import MODNet


def load_modnet(ckpt_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load the pre-trained MODNet model."""
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if device == 'cuda':
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))

    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet


def process_image(image_path):
    """Read and preprocess a single image for MODNet inference."""
    im = Image.open(image_path)

    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    im = Image.fromarray(im)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    im = transform(im)
    im = im[None, :, :, :]  

    return im


def resize_image(image_tensor, ref_size=512):
    """Resize the image tensor to the reference size."""
    im_b, im_c, im_h, im_w = image_tensor.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw -= im_rw % 32
    im_rh -= im_rh % 32
    return F.interpolate(image_tensor, size=(im_rh, im_rw), mode='area')


def infer_matte(modnet, image_tensor, original_size):
    """Run MODNet inference on a single image and return the matte."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, matte = modnet(image_tensor.to(device), True)

    im_h, im_w = original_size
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return (matte * 255).astype('uint8')


def save_matte(matte, output_path):
    """Save the matte image."""
    matte_image = Image.fromarray(matte, mode='L')
    matte_image.save(output_path)


def threshold_matte(matte, threshold=128):
    """Apply a threshold to the matte to reduce blur at edges."""
    matte[matte < threshold] = 0
    matte[matte >= threshold] = 255
    return matte


def sharpen_matte(matte):
    """Apply sharpening filter to the matte to reduce blur around edges."""
    matte_image = Image.fromarray(matte, mode='L')
    matte_image = matte_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return np.array(matte_image)


def reduce_blur(matte, radius=1):
    """Apply a Gaussian blur to reduce the blur radius."""
    matte_image = Image.fromarray(matte, mode='L')
    matte_image = matte_image.filter(ImageFilter.GaussianBlur(radius))
    return np.array(matte_image)


def refine_matte(matte, method='threshold', threshold=128, radius=1):
    """Refine the matte by applying thresholding, sharpening, or Gaussian blur."""
    if method == 'threshold':
        matte = threshold_matte(matte, threshold)
    elif method == 'sharpen':
        matte = sharpen_matte(matte)
    elif method == 'blur':
        matte = reduce_blur(matte, radius)
    return matte


def get_matte_from_modnet(image_path, output_path, ckpt_path, refinement_method='threshold', threshold=128, blur_radius=1):
    """Process a single image and save the refined matte output."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Cannot find input image: {image_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint path: {ckpt_path}")

    modnet = load_modnet(ckpt_path)

    im = process_image(image_path)

    im_resized = resize_image(im)

    matte = infer_matte(modnet, im_resized, im.shape[2:])

    matte = refine_matte(matte, method=refinement_method, threshold=threshold, radius=blur_radius)

    save_matte(matte, output_path)

    print(f"Processed and saved matte to {output_path}")


