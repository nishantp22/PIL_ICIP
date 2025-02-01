from diffusers import FluxPipeline, FluxImg2ImgPipeline, AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from transformers import T5EncoderModel, T5TokenizerFast, CLIPTokenizer, CLIPTextModel
import torch
import numpy as np
import gc
from PIL import Image
import torchvision.transforms as transforms


print(torch.cuda.device_count())

device = torch.device('cuda')

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
# def run_rmbg(img, sigma=0.0):
#     H, W, C = img.shape
#     assert C == 3
#     k = (256.0 / float(H * W)) ** 0.5
#     feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
#     feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
#     alpha = rmbg(feed)[0][0]
#     alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
#     alpha = alpha.movedim(1, -1)[0]
#     alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
#     result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
#     return result.clip(0, 255).astype(np.uint8), alpha


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def process(prompt, highres_scale, steps, highres_denoise, height, width):

    flush()

    ckpt_id = "black-forest-labs/FLUX.1-schnell"

    text_encoder = CLIPTextModel.from_pretrained(
        ckpt_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        ckpt_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
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
    print(transformer.dtype)

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
        guidance_scale=0.0,
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
        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")
        image[0].save("image.png")

    img=image[0]   

    pixels = transforms.ToTensor()(image[0])

    pixels = pixels.unsqueeze(0)  # Shape becomes [1, C, W, H]
    pixels = pixels.to(device =device , dtype = torch.bfloat16)
    latentsa = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    print(latentsa.dtype)
    # latentsa = latentsa.to(dtype=torch.bfloat16)

    # print(pixels.shape)
    # pixels=image[0]

    pixels = pytorch2numpy(pixels)


    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(width * highres_scale / 64.0) * 64),
        target_height=int(round(height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=device)


    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

        # fg = resize_and_center_crop(input_fg, image_width, image_height)
        # concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
        # concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    del vae

    flush()


    transformer = FluxTransformer2DModel.from_pretrained(ckpt_id, subfolder="transformer", torch_dtype=torch.bfloat16)

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
    
    latents = i2i_pipe(
        image=img,
        strength=highres_denoise,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        width=image_width,
        latents=latentsa,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        # num_images_per_prompt=num_samples,
            # generator=rng,
        output_type='latent',
        # guidance_scale=cfg,
            # joint_attention_kwargs={'concat_conds': concat_conds},
    ).images

    del pipeline.transformer
    del pipeline
    del transformer

    flush()

    vae = AutoencoderKL.from_pretrained(ckpt_id, revision="refs/pr/1", subfolder="vae", torch_dtype=torch.bfloat16).to(
        "cuda"
    )

    vae_scale_factor = 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor*2)

    with torch.no_grad():
        print("Running decoding.")
        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")
        image[0].save("imageF.png")

process("a 21 year old indian man standing in a football stadium with bright sunlight falling on his face from right", 1.5, 25, 0.5, 512, 512)