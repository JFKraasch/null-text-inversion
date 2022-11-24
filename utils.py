import numpy as np
from PIL import Image
import torch


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def show_lat(latents, pipe):
    # utility function for visualization of diffusion process
    with torch.no_grad():
        images = pipe.decode_latents(latents)
        print("Image statistics: ", images.mean(), images.std(), images.min(), images.max())
        im = pipe.numpy_to_pil(images)[0].resize((128, 128))
    return im


@torch.no_grad()
def reconstruct(pipe, z_T, prompt, null_text_embeddings, T, w=7.5):
    with torch.inference_mode(), torch.autocast("cuda"):
        text_embeddings = pipe._encode_prompt(prompt, pipe.device, 1, False, None)
        latents = z_T.to(pipe.device)
        pipe.scheduler.set_timesteps(T)
        for i, (t, null_text_t) in enumerate(pipe.progress_bar(zip(pipe.scheduler.timesteps, null_text_embeddings))):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            input_embedding = torch.cat([null_text_t.to(pipe.device), text_embeddings])
            # predict the noise residual
            print(latent_model_input.shape, input_embedding.shape)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + w * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1

            g = torch.Generator(device=pipe.device).manual_seed(84)
            latents = pipe.scheduler.step(noise_pred, t, latents, generator=g).prev_sample

        #Post-processing
        image = pipe.decode_latents(latents)
        return image