import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

from p2p import layers
from p2p import display_utils
from p2p.display_utils import run_and_display, show_cross_attention
from p2p.layers import AttentionReplace, AttentionRefine, LocalBlend, get_equalizer, AttentionReweight

project_name = "baby_blond"
device = 'cuda'
model_id_or_path = "runwayml/stable-diffusion-v1-5"
token = "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"
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path,
                                               scheduler=DDIMScheduler.from_config(model_id_or_path,
                                                                                   subfolder="scheduler",
                                                                                   use_auth_token=token),
                                               use_auth_token=token).to("cuda")
generator = torch.Generator()

init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")
z_T = init_trajectory[-1]
prompts = ["A baby wearing a blue shirt lying on the sofa"]
controller = layers.AttentionStore()
image, _ = run_and_display(pipe, prompts, controller, latent=z_T, null_prompts=null_embeddings, run_baseline=False, generator=generator, T=50, w=7.5)
show_cross_attention(pipe, prompts, controller, res=16, from_where=("up", "down"))

prompts = ["A baby wearing a blue shirt lying on the sofa",
           "A baby wearing a blue shirt sitting on the sofa"]

controller = AttentionReplace(pipe, prompts, 50, cross_replace_steps=.8, self_replace_steps=0.8)
_ = run_and_display(pipe, prompts, controller, latent=z_T, null_prompts=null_embeddings,run_baseline=True,  generator=generator,fp="editedp2p.png", T=50, w=7.5)


prompts = ["A baby wearing a blue shirt lying on the sofa",
           "A baby wearing a blue shirt sitting on the sofa"]

### pay 3 times more attention to the word "smiling"
equalizer = get_equalizer(pipe, prompts[1], ("sitting",), (5,))
controller = AttentionReweight(pipe, prompts, 50, cross_replace_steps=.8,
                               self_replace_steps=.8,
                               equalizer=equalizer)
_ = run_and_display(pipe, prompts, controller, latent=z_T, null_prompts=null_embeddings,run_baseline=True,  generator=generator,fp="refinedp2p.png", T=50, w=7.5)