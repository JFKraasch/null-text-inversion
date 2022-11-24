from utils import preprocess, show_lat
import torch
import numpy as np

def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image

@torch.no_grad()
def ddim_inversion(pipe, prompt, image, T, generator=None, negative_prompt="", w=1):
    """
    DDIM based inversion of image to noise

    :param pipe: Diffusion Pipeline
    :param prompt: initial prompt
    :param image: input image that should be inversed
    :param T: num_steps of Diffusion
    :param generator: noise generator
    :param negative_prompt: negative prompt for guidance
    :param w: guidance scale
    :return: initial trajectory
    """
    pp_image = preprocess_image(image)
    latents = pipe.vae.encode(pp_image.to(pipe.device)).latent_dist.sample(generator=generator) * 0.18215


    context = pipe._encode_prompt(prompt, pipe.device, 1, False, negative_prompt)
    pipe.scheduler.set_timesteps(T)

    next_latents = latents
    all_latents = [latents.detach().cpu().unsqueeze(0)]

    for timestep, next_timestep in zip(reversed(pipe.scheduler.timesteps[1:]),
                                       reversed(pipe.scheduler.timesteps[:-1])):
        latent_model_input = pipe.scheduler.scale_model_input(next_latents, timestep)
        noise_pred = pipe.unet(latent_model_input, timestep, context).sample

        alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_next = pipe.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        f = (next_latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        next_latents = alpha_prod_t_next ** 0.5 * f + beta_prod_t_next ** 0.5 * noise_pred
        all_latents.append(next_latents.detach().cpu().unsqueeze(0))

    return torch.cat(all_latents)

if __name__ == "__main__":
    from pathlib import Path

    import torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline
    from diffusers.schedulers import DDIMScheduler
    from matplotlib import pyplot as plt

    project_name = "room"
    Path(f"./results/{project_name}").mkdir(parents=True, exist_ok=True)

    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    token = "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"
    SD_pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path,
                                                      scheduler=DDIMScheduler.from_config(model_id_or_path,
                                                                              subfolder="scheduler",
                                                                              use_auth_token=token),
                                                      use_auth_token=token).to("cuda")



    og_image = Image.open("./scratch/test2.png").resize((512, 512))
    source_prompt = "((Side view)) of an empty class room with glass windows and wooden floor, purple neon lighting, anime"
    T = 50
    generator = torch.Generator(device="cuda")
    init_trajectory = ddim_inversion(SD_pipe, source_prompt, og_image, T, generator)
    print(init_trajectory.shape)
    torch.save(init_trajectory, f"./results/{project_name}/init_trajectory.pt")

    plt.figure(figsize=(20, 8))
    with torch.autocast("cuda"):
        for i, traj in enumerate(init_trajectory[::10]):
            plt.subplot(1, (T // 10) + 1, i + 1)
            plt.imshow(show_lat(traj.to("cuda"), SD_pipe))
            plt.axis("off")
    plt.savefig(f"./results/{project_name}/trajectories.png")

    with torch.inference_mode(), torch.autocast("cuda"):
        z_T = init_trajectory[-1].to("cuda")
        im = SD_pipe(prompt=source_prompt, latents=z_T, generator=generator)
        im[0][0].save(f"./results/{project_name}/DDIM_reconstruction.png")


