import io

import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt

from ddim_inversion import ddim_inversion
from null_edit import null_text_inversion
from reconstruct import reconstruct

from pathlib import Path
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler

from dotenv import load_dotenv
import os

from flask import Flask, Response, request

from utils import show_lat

load_dotenv()

app = Flask(__name__)


@app.route('/DDIM', methods=['POST', 'OPTIONS'])
def ddim_process():
    if (request.method == 'OPTIONS'):
        print('got options 1')
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        print('got options 2')
        return response
    yaml_file = request.form.get('config')
    print(yaml_file)
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)
    project_name = config['project_name']
    source_prompt = config["prompt"]
    og_image = Image.open(config["image"]).resize((512, 512))
    T = config["steps"]
    Path(f"./results/{project_name}").mkdir(parents=True, exist_ok=True)
    init_trajectory = ddim_inversion(pipe, source_prompt, og_image, T, generator)
    torch.save(init_trajectory, f"./results/{project_name}/init_trajectory.pt")

    response = Response(f"./results/{project_name}/init_trajectory.pt")
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Expose-Headers'] = '*'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    return response

@app.route('/inversion', methods=['POST', 'OPTIONS'])
def inversion():
    if (request.method == 'OPTIONS'):
        print('got options 1')
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        print('got options 2')
        return response

    yaml_file = request.form.get('config')
    print(yaml_file)
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    project_name = config['project_name']
    source_prompt = config["prompt"]
    T = config["steps"]
    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
    except Exception as e:
        return e, 400

    _, null_embeddings = null_text_inversion(pipe, init_trajectory, source_prompt,
                                               guidance_scale=7.5, generator=generator)
    torch.save(null_embeddings, f"./results/{project_name}/nulls.pt")

    response = Response(f"./results/{project_name}/nulls.pt")
    response.headers['Content-Type'] = 'text/plain'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Expose-Headers'] = '*'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    return response

@app.route('/edit', methods=['POST', 'OPTIONS'])
def edit():
    if (request.method == 'OPTIONS'):
        print('got options 1')
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        print('got options 2')
        return response

    yaml_file = request.form.get('config')
    print(yaml_file)
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    project_name = config['project_name']
    edited_prompt = config["edited_prompt"]
    T = config["steps"]
    guidance_scale = config["guidance_scale"]
    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")
        z_T = init_trajectory[-1]
    except Exception as e:
        return e, 400

    edited_img = reconstruct(pipe, z_T, edited_prompt, null_embeddings, guidance_scale=guidance_scale, T=T)
    buf = io.BytesIO()
    print(edited_img.shape)
    im = Image.fromarray(np.uint8(edited_img[0]*255)).convert('RGB')
    im.save(f"./results/{project_name}/edited.png", format="PNG")
    im.save(buf, format='JPEG')
    response = Response(buf.getvalue())
    response.headers['Content-Type'] = 'image/jpeg'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Expose-Headers'] = '*'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    return response

@app.route('/ddim_recon', methods=['POST', 'OPTIONS'])
def ddim_recon():
    if (request.method == 'OPTIONS'):
        print('got options 1')
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        print('got options 2')
        return response

    yaml_file = request.form.get('config')
    print(yaml_file)
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    project_name = config['project_name']
    source_prompt = config["prompt"]
    T = config["steps"]
    guidance_scale = config["guidance_scale"]
    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        z_T = init_trajectory[-1]
    except Exception as e:
        return e, 400

    with torch.inference_mode(), torch.autocast("cuda"):
        z_T = init_trajectory[-1].to("cuda")
        im = pipe(prompt=source_prompt, latents=z_T, generator=generator)
        im[0][0].save(f"./results/{project_name}/DDIM_reconstruction.png")

        buf = io.BytesIO()
        im[0][0].convert("RGB").save(buf, format='JPEG')
        response = Response(buf.getvalue())
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        return response



@app.route('/guidance', methods=['POST', 'OPTIONS'])
def guidance_test():
    if (request.method == 'OPTIONS'):
        print('got options 1')
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        response.headers['Access-Control-Allow-Methods'] = '*'
        response.headers['Access-Control-Expose-Headers'] = '*'
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        print('got options 2')
        return response

    yaml_file = request.form.get('config')
    print(yaml_file)
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    project_name = config['project_name']
    source_prompt = config["prompt"]
    edited_prompt = config["edited_prompt"]
    num_guidance = config["num_guidance"]
    T = config["steps"]
    guidance_scale = config["guidance_scale"]
    min_guidance = config["min_guidance"]
    max_guidance = config["max_guidance"]
    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")
        z_T = init_trajectory[-1]
    except Exception as e:
        return e, 400

    recon_img = reconstruct(pipe, z_T, source_prompt, null_embeddings, guidance_scale=guidance_scale, T=T)
    edit_imgs = []
    for scale in np.linspace(0.5, 10, num_guidance):
        edit_img = reconstruct(pipe, z_T, edited_prompt, null_embeddings, guidance_scale=scale)
        edit_imgs.append(edit_img)

    fig, ax = plt.subplots(1, num_guidance + 1, figsize=(10 * (num_guidance + 1), 10))

    ax[0].imshow(recon_img[0])
    ax[0].set_title("Reconstructed", fontdict={'fontsize': 40})
    ax[0].axis('off')

    for i, scale in enumerate(np.linspace(min_guidance, max_guidance, num_guidance)):
        ax[i + 1].imshow(edit_imgs[i][0])
        ax[i + 1].set_title("%.2f" % scale, fontdict={'fontsize': 40})
        ax[i + 1].axis('off')
    plt.xlabel(edited_prompt)

    # Saving the figure
    plt.savefig(f"./results/{project_name}/guidance_test.png")
    im = Image.open(f"./results/{project_name}/guidance_test.png").convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format='JPEG')
    response = Response(buf.getvalue())
    response.headers['Content-Type'] = 'image/jpeg'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Expose-Headers'] = '*'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    return response



if __name__ == '__main__':
    device = 'cuda'
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    token = "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"
    pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path,
                                                   scheduler=DDIMScheduler.from_config(model_id_or_path,
                                                                                       subfolder="scheduler",
                                                                                       use_auth_token=token),
                                                   use_auth_token=token).to("cuda")
    generator = torch.Generator(device="cuda")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
