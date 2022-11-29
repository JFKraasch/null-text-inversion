from typing import List

import numpy as np
import torch
from PIL import Image

from p2p import ptp_utils
from p2p.layers import AttentionStore, EmptyControl, LOW_RESOURCE


def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(pipe,prompts,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, fp="attention.png"):
    tokens = pipe.tokenizer.encode(prompts[select])
    decoder = pipe.tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return ptp_utils.view_images(np.stack(images, axis=0), fp=fp)


def show_self_attention_comp(prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=10, select: int = 0, fp="attention.png"):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1), fp=fp)

def run_and_display(pipe, prompts, controller, latent=None, null_prompts=None,run_baseline=False, generator=None, T=50, w=7.5, fp="test.png"):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(pipe, prompts, EmptyControl(), latent=latent, null_prompts=null_prompts, run_baseline=False, generator=generator, fp="baseline.png", T=T)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(pipe, prompts, controller, latent=latent, null_prompts=null_prompts, num_inference_steps=T,
                                                  guidance_scale=w, generator=generator, low_resource=LOW_RESOURCE)
    pil_image = ptp_utils.view_images(images, fp=fp)
    return images, x_t, pil_image