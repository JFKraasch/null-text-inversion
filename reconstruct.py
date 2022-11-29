import numpy as np
import torch
from tqdm import tqdm

@torch.no_grad()
def reconstruct(pipe, latents, prompt, null_text_embeddings, guidance_scale=7.5, generator=None, eta=0.0, negative_prompt="", T=50):
    text_embeddings = pipe._encode_prompt(prompt, pipe.device, 1, False, negative_prompt)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    latents = latents.to(pipe.device)

    pipe.scheduler.set_timesteps(T)
    print(len(null_text_embeddings))
    for i, (t, null_text_t) in enumerate(pipe.progress_bar(zip(pipe.scheduler.timesteps, null_text_embeddings))):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        input_embedding = torch.cat([null_text_t.to(pipe.device), text_embeddings])
        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    #Post-processing
    image = pipe.decode_latents(latents)
    return image

if __name__ == "__main__":
    import torch
    from diffusers import StableDiffusionPipeline
    from matplotlib import pyplot as plt
    from diffusers.schedulers import DDIMScheduler
    from pathlib import Path


    project_name = "room"
    Path(f"./results/{project_name}").mkdir(parents=True, exist_ok=True)

    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    token = "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"
    SD_pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path,
                                                      scheduler=DDIMScheduler.from_config(model_id_or_path,
                                                                              subfolder="scheduler",
                                                                              use_auth_token=token),
                                                      use_auth_token=token).to("cuda")

    T = 50
    source_prompt = "((Side view)) of an empty class room with glass windows and wooden floor, purple neon lighting, anime"
    init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
    generator = torch.Generator(device="cuda")
    z_T = init_trajectory[-1]

    null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")

    SD_pipe.scheduler.set_timesteps(T)
    recon_img = reconstruct(SD_pipe, z_T, source_prompt, null_embeddings, guidance_scale=1)
    plt.imsave(f"./results/{project_name}/reconstructed.png", recon_img[0])

    edited_prompt = "((Top down view)) of an empty class room with glass windows and wooden floor, purple neon lighting, anime"
    edited_img = reconstruct(SD_pipe, z_T, edited_prompt, null_embeddings, guidance_scale=7.5)
    plt.imsave(f"./results/{project_name}/edited.png", edited_img[0])

    edit_imgs = []
    num_imgs = 10
    for scale in np.linspace(0.5, 10, num_imgs):
        edit_img = reconstruct(SD_pipe, z_T, edited_prompt, null_embeddings, guidance_scale=scale)
        edit_imgs.append(edit_img)

    fig, ax = plt.subplots(1, num_imgs + 1, figsize=(10 * (num_imgs + 1), 10))

    ax[0].imshow(recon_img[0])
    ax[0].set_title("Reconstructed", fontdict={'fontsize': 40})
    ax[0].axis('off')

    for i, scale in enumerate(np.linspace(0.5, 10, num_imgs)):
        ax[i + 1].imshow(edit_imgs[i][0])
        ax[i + 1].set_title("%.2f" % scale, fontdict={'fontsize': 40})
        ax[i + 1].axis('off')

    plt.xlabel(edited_prompt)
    plt.savefig(f"./results/{project_name}/guidance_test.png")
