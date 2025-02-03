from diffusers import FluxPipeline, FluxImg2ImgPipeline, AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel
import torch
import numpy as np
import gc
from PIL import Image
import torchvision.transforms as transforms
import random
import cv2
from briarmbg import BriaRMBG

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda')


rng = torch.Generator(device=device).manual_seed(int(random.randrange(100000000)))


num_channels_latents = 16

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)


    return latents

@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h

def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(rmbg, img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def process(prompt, highres_scale, steps, highres_denoise, height, width):

    ckpt_id = "black-forest-labs/FLUX.1-schnell"

    rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4").to("cuda")
    vae = AutoencoderKL.from_pretrained(ckpt_id, revision="refs/pr/1", subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")


    fg=cv2.imread("images/original_image.png")
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)

    fg, matting = run_rmbg(rmbg, fg)

    fg = resize_and_center_crop(fg, width, height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    # concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    del rmbg
    del vae

    flush()


    text_encoder = CLIPTextModel.from_pretrained(ckpt_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder_2 = T5EncoderModel.from_pretrained(ckpt_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    tokenizer = CLIPTokenizer.from_pretrained(ckpt_id, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(ckpt_id, subfolder="tokenizer_2")

    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=None,
        vae=None,
    ).to("cuda")

    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )

    del text_encoder
    del text_encoder_2
    del tokenizer
    del tokenizer_2
    del pipeline

    flush()

    transformer = FluxTransformer2DModel.from_pretrained(ckpt_id, subfolder="transformer", torch_dtype=torch.bfloat16)

    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        vae=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print("Running denoising.")
    latents = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=25,
        guidance_scale=3.5,
        height=height,
        width=width,
        output_type="latent",
    ).images
    print(f"{latents.shape=}")

    del pipeline.transformer
    del pipeline
    del transformer

    flush()

    vae = AutoencoderKL.from_pretrained(ckpt_id, revision="refs/pr/1", subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")

    vae_scale_factor = 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor*2)

    with torch.no_grad():
        print("Running decoding.")
        latentsb = latents
        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")
        image[0].save("image.png")
        pixels = transforms.ToTensor()(image[0])
        pixels = pixels.unsqueeze(0)  # Shape becomes [1, C, W, H]

        pixels = pixels.to(device =device , dtype = torch.bfloat16)
        latentsa = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
        img=image[0]   


    del vae
    del image_processor
    # del latents

    flush()


    # pixels = pytorch2numpy(pixels)


    # pixels = [resize_without_crop(
    #     image=p,
    #     target_width=int(round(width * highres_scale / 64.0) * 64),
    #     target_height=int(round(height * highres_scale / 64.0) * 64))
    # for p in pixels]

    # pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)

    # latents = latents.to(device=device)


    image_height, image_width = latentsa.shape[2] * 8, latentsa.shape[3] * 8


        # fg = resize_and_center_crop(input_fg, image_width, image_height)
        # concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
        # concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor




    transformer = FluxTransformer2DModel.from_pretrained(ckpt_id, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda")

    i2i_pipe = FluxImg2ImgPipeline.from_pretrained(
        ckpt_id,
        vae=None,
        text_encoder_2=None,
        tokenizer_2=None,
        text_encoder=None,
        transformer=transformer,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print(latentsb.shape)

    print("Running i2i denoising.")
    latents = i2i_pipe(
        image=img,
        strength=highres_denoise,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        width=image_width,
        latents=latentsb,
        guidance_scale=3.5,
        height=image_height,
        num_inference_steps=50,
        generator=rng,
        output_type='latent',
        # guidance_scale=cfg,
        # joint_attention_kwargs={'concat_conds': concat_conds},
    ).images

    del i2i_pipe.transformer
    del i2i_pipe
    del transformer

    flush()

    vae = AutoencoderKL.from_pretrained(ckpt_id, revision="refs/pr/1", subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")

    vae_scale_factor = 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor*2)

    with torch.no_grad():
        print("Running decoding.")
        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")
        image[0].save("imageF.png")

process("a cat holding a sign that says hello world", 1.5, 25, 0.5, 1024, 1024)