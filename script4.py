#using flux over sd15 without iclight weights

import os
import math
import numpy as np
import torch
import safetensors.torch as sf

from PIL import Image
from transformers import T5Tokenizer
from transformers import T5EncoderModel
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from diffusers import FluxTransformer2DModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer, T5Tokenizer
from transformers import BertModel
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
import random
from diffusers import FlowMatchEulerDiscreteScheduler
import cv2
import inspect






# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

ckpt_id = "black-forest-labs/FLUX.1-dev"
ckpt_4bit_id = "hf-internal-testing/flux.1-dev-nf4-pkg"


transformer = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer")
text_encoder_2= T5EncoderModel.from_pretrained(ckpt_4bit_id, subfolder="text_encoder_2")
# transformer = FluxTransformer2DModel.from_pretrained(ckpt_id, subfolder="transformer")
# text_encoder_2= T5EncoderModel.from_pretrained(ckpt_id, subfolder="text_encoder_2")





tokenizer = CLIPTokenizer.from_pretrained(ckpt_id, subfolder="tokenizer") 
text_encoder = CLIPTextModel.from_pretrained(ckpt_id, subfolder="text_encoder")
sd15_name = 'black-forest-labs/FLUX.1-dev'
tokenizer_2 = T5Tokenizer.from_pretrained(ckpt_id, subfolder="tokenizer_2")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")


transformer_original_forward = transformer.forward


# def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
#     c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
#     c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
#     new_sample = torch.cat([sample, c_concat], dim=1)
#     kwargs['cross_attention_kwargs'] = {}
#     return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


# unet.forward = hooked_unet_forward


'''
    ToDO: Modify the hooked_unet_forward method to configure it with the transformer instead. The code below is not working because the sample parameter
    from UNetConditional2DModel of dimension 4 [batch size, img width, img height, channels] but the transformer from 
    FluxTransformer2DModel instead has a hidden state parameter with 
    dimension 3 [batch size, image_sequence_length]
'''

def hooked_transformer_forward(hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance, **kwargs):
    # print(kwargs)
    c_concat = kwargs['joint_attention_kwargs']['concat_conds'].to(hidden_states)

    batch_size, channels, width, height = c_concat.shape
    c_concat = c_concat.reshape(batch_size, width * height, channels) 
    c_concat = c_concat.reshape(batch_size, hidden_states.shape[1] , hidden_states.shape[2])

    c_concat = torch.cat([c_concat] * (hidden_states.shape[0] // c_concat.shape[0]), dim=0)

    new_sample = torch.cat([hidden_states, c_concat], dim=1)
    kwargs['joint_attention_kwargs'] = {}

    return transformer_original_forward(new_sample, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance, **kwargs)



transformer.forward = hooked_transformer_forward

# Load

# model_path = './models/iclight_sd15_fc.safetensors'

# if not os.path.exists(model_path):
#     download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

# sd_offset = sf.load_file(model_path)
# sd_origin = unet.state_dict()

# print(sd_origin.keys())

# print("Original model keys:", len(sd_origin.keys()))
# print("Safetensors model keys:", len(sd_offset.keys()))
# keys = sd_origin.keys()
# sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
# unet.load_state_dict(sd_merged, strict=True)
# del sd_offset, sd_origin, sd_merged, keys

# # Device

device = torch.device('cuda')
text_encoder_2 = text_encoder_2.to(device=device)
text_encoder=text_encoder.to(device=device)
vae = vae.to(device=device, dtype=torch.bfloat16)
transformer = transformer.to(device=device)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# # SDP

transformer.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# # Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

flow_match_euler_scheduler = FlowMatchEulerDiscreteScheduler()

# # Pipelines

t2i_pipe = FluxPipeline(
    vae=vae,
    text_encoder_2=text_encoder_2,
    text_encoder=None,
    tokenizer_2=tokenizer_2,
    tokenizer=None,
    transformer=transformer,
    scheduler=flow_match_euler_scheduler,
)

i2i_pipe = FluxImg2ImgPipeline(
    vae=vae,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    text_encoder=None,
    transformer=transformer,
    tokenizer=None,
    scheduler=flow_match_euler_scheduler,
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer_2.model_max_length
    chunk_length = tokenizer_2.model_max_length - 2
    id_start = tokenizer_2.bos_token_id or tokenizer_2.convert_tokens_to_ids("<s>")
    id_end = tokenizer_2.eos_token_id or tokenizer_2.convert_tokens_to_ids("</s>")
    id_pad = tokenizer_2.pad_token_id or tokenizer_2.convert_tokens_to_ids("<pad>")

    tokens = tokenizer_2(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))
    
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]
    token_ids = torch.tensor(chunks, dtype=torch.long).to(device)



    outputs = text_encoder_2(token_ids)


    prompt_embeds = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]



    prompt = [txt] if isinstance(txt, str) else txt
    batch_size = len(prompt)


    text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])

    pooled_prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    pooled_prompt_embeds = pooled_prompt_embeds.pooler_output
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * 1, -1)
    
    return prompt_embeds, pooled_prompt_embeds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c, c_pooled = encode_prompt_inner(positive_prompt)
    uc, uc_pooled = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return (c, c_pooled), (uc, uc_pooled)



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
def run_rmbg(img, sigma=0.0):
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


@torch.inference_mode()
def process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    bg_source = BGSource(bg_source)
    input_bg = None

    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)

    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    # conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)
    (conds, conds_pooled), (unconds, unconds_pooled) = encode_prompt_pair(
    positive_prompt=prompt + ', ' + a_prompt,
    negative_prompt=n_prompt,
)

    if input_bg is None:
        latents = t2i_pipe(
        prompt_embeds=conds,
        pooled_prompt_embeds=conds_pooled,
        negative_prompt_embeds=unconds,
        negative_pooled_prompt_embeds=unconds_pooled,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            joint_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            joint_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=transformer.device, dtype=transformer.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        joint_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample

    px = pytorch2numpy(pixels)

    return px


@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg)
    results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    img = Image.fromarray(results[0])
    img.save("relighted_image.png")
    img2 = Image.fromarray(input_fg)
    img2.save("input_fg.png")
    return input_fg, results


quick_prompts = [
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm'
]
quick_prompts = [[x] for x in quick_prompts]


quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]
quick_subjects = [[x] for x in quick_subjects]


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"



def apply_relight(input_fg):

    fg=cv2.imread(input_fg)
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)

    prompt = "a man standing in a park in bright sunlight"
    bg_source = BGSource.NONE.value
    num_samples = 1
    seed = random.randrange(100000000)
    image_width=fg.shape[1]
    image_height = fg.shape[0]
    steps = 25
    cfg = 2
    highres_scale = 1.5
    highres_denoise = 0.5
    lowres_denoise = 0.9
    a_prompt = 'best quality'
    n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'

    process_relight(fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)


apply_relight("images/original_image.png")