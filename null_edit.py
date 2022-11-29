import torch

def null_text_inversion(
        pipe,
        all_latents,
        prompt,
        num_opt_steps=10,
        lr=0.01,
        tol=1e-5,
        guidance_scale=7.5,
        eta: float = 0.0,
        generator=None,
        T=50,
        negative_prompt=None
):
    # get null text embeddings for prompt
    null_text_prompt = ""
    null_text_input = pipe.tokenizer(
        null_text_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    null_text_embeddings = torch.nn.Parameter(pipe.text_encoder(null_text_input.input_ids.to(pipe.device))[0],
                                              requires_grad=True)
    null_text_embeddings = null_text_embeddings.detach()
    null_text_embeddings.requires_grad_(True)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        [null_text_embeddings],  # only optimize the embeddings
        lr=lr,
    )

    # step_ratio = pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    text_embeddings = pipe._encode_prompt(prompt, pipe.device, 1, False, negative_prompt).detach()
    # input_embeddings = torch.cat([null_text_embeddings, text_embeddings], dim=0)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    all_null_texts = []
    latents = all_latents[-1]
    latents = latents.to(pipe.device)

    pipe.scheduler.set_timesteps(T)
    for timestep, prev_latents in pipe.progress_bar(zip(pipe.scheduler.timesteps, reversed(all_latents[:-1]))):
        prev_latents = prev_latents.to(pipe.device).detach()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = pipe.scheduler.scale_model_input(latents, timestep).detach()
        noise_pred_text = pipe.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample.detach()

        for idx in range(num_opt_steps):
            # predict the noise residual
            noise_pred_uncond = pipe.unet(latent_model_input, timestep,
                                          encoder_hidden_states=null_text_embeddings).sample

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            prev_latents_pred = pipe.scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample
            loss = torch.nn.functional.mse_loss(prev_latents_pred, prev_latents).mean()

            loss.backward()
            print(idx, loss, null_text_embeddings.grad.mean())
            optimizer.step()
            optimizer.zero_grad()
            if loss < tol:
                break
        all_null_texts.append(null_text_embeddings.detach().cpu().unsqueeze(0))
        latents = prev_latents_pred.detach()
    return all_latents[-1], torch.cat(all_null_texts)


if __name__ == "__main__":
    from pathlib import Path
    from diffusers import StableDiffusionPipeline
    from diffusers.schedulers import DDIMScheduler

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
    z_T, null_embeddings = null_text_inversion(SD_pipe, init_trajectory, source_prompt,
                                               guidance_scale=7.5, generator=generator)

    torch.save(null_embeddings, f"./results/{project_name}/nulls.pt")
