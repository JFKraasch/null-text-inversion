import io
import copy
import numpy as np
import torch
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename

from ddim_inversion import ddim_inversion
from null_edit import null_text_inversion
from p2p import layers
from p2p.display_utils import run_and_display, show_cross_attention
from p2p.layers import AttentionReplace, AttentionRefine
from reconstruct import reconstruct

from pathlib import Path
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler

from dotenv import load_dotenv
import os

from flask import Flask, Response, request, flash, redirect, url_for

from utils import show_lat

load_dotenv()
UPLOAD_FOLDER = './scratch'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

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

    try:
        yaml_file = request.form.get('config')
        print(yaml_file)
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            filename = config["image"]
            T = config["steps"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            filename = request.form.get("image")
            T = int(request.form.get("steps"))
            og_image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).resize((512, 512))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

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

    try:
        yaml_file = request.form.get('config')
        print(yaml_file)
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            T = config["steps"]
            num_opt_steps = config["N"]
    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            T = int(request.form.get("steps"))
            num_opt_steps = int(request.form.get("N"))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
    except Exception as e:
        return e, 400
    print("num_opt_stpes", num_opt_steps)
    _, null_embeddings = null_text_inversion(pipe, init_trajectory, source_prompt,
                                               guidance_scale=7.5, generator=generator, num_opt_steps=num_opt_steps, T=T)
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

@app.route('/naive_edit', methods=['POST', 'OPTIONS'])
def naive_edit():
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

    try:
        yaml_file = request.form.get('config')
        print(yaml_file)
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            edited_prompt = config["edited_prompt"]
            guidance_scale = config["guidance_scale"]
            T = config["steps"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            edited_prompt = request.form.get("edited_prompt")
            guidance_scale = float(request.form.get("guidance_scale"))
            T = int(request.form.get("steps"))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

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

@app.route('/p2p_replace', methods=['POST', 'OPTIONS'])
def p2p_replace():
    """
        Call prompt to prompt AttentionReplace controller given config file
        Important num of words in source and edited prompt need to be the same
        :return:
        """
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

    try:
        yaml_file = request.form.get('config')
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            edited_prompt = config["edited_prompt"]
            guidance_scale = config["guidance_scale"]
            T = config["steps"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            edited_prompt = request.form.get("edited_prompt")
            guidance_scale = float(request.form.get("guidance_scale"))
            T = int(request.form.get("steps"))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")
        z_T = init_trajectory[-1]
    except Exception as e:
        return e, 400
    prompts = [source_prompt, edited_prompt]
    p2p_pipe = copy.deepcopy(pipe)#.clone()
    controller = AttentionReplace(p2p_pipe, prompts, T, cross_replace_steps=.8, self_replace_steps=0.4)
    _, _, edited_image = run_and_display(p2p_pipe, prompts, controller, latent=z_T, null_prompts=null_embeddings, run_baseline=False,
                    generator=generator, fp=f"./results/{project_name}/p2p_replace.png", T=T, w=guidance_scale)

    buf = io.BytesIO()
    edited_image.save(buf, format='JPEG')
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


@app.route('/p2p_refine', methods=['POST', 'OPTIONS'])
def p2p_refine():
    """
    Call prompt to prompt AttentionRefine controller given config file
    :return:
    """
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

    try:
        yaml_file = request.form.get('config')
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            edited_prompt = config["edited_prompt"]
            guidance_scale = config["guidance_scale"]
            T = config["steps"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            edited_prompt = request.form.get("edited_prompt")
            guidance_scale = float(request.form.get("guidance_scale"))
            T = int(request.form.get("steps"))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")
        z_T = init_trajectory[-1]
    except Exception as e:
        return e, 400

    prompts = [source_prompt, edited_prompt]

    p2p_pipe = copy.deepcopy(pipe)
    controller = AttentionRefine(p2p_pipe, prompts, T, cross_replace_steps=.8, self_replace_steps=0.4)
    _, _, edited_image = run_and_display(p2p_pipe, prompts, controller, latent=z_T, null_prompts=null_embeddings,
                                         run_baseline=False,
                                         generator=generator, fp=f"./results/{project_name}/p2p_refine.png", T=T,
                                         w=guidance_scale)

    buf = io.BytesIO()
    edited_image.save(buf, format='JPEG')
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

@app.route('/p2p_evaluate', methods=['POST', 'OPTIONS'])
def p2p_evaluate():
    """
        Generates image based on source prompt and null-text inversion and extracts attention based on prompt
        Should reconstruct original image
        :return:
        """
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

    try:
        yaml_file = request.form.get('config')
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            guidance_scale = config["guidance_scale"]
            T = config["steps"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            guidance_scale = float(request.form.get("guidance_scale"))
            T = int(request.form.get("steps"))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")
        z_T = init_trajectory[-1]
    except Exception as e:
        return e, 400
    prompts = [source_prompt]
    p2p_pipe = copy.deepcopy(pipe)
    controller = layers.AttentionStore()
    image, _, _ = run_and_display(p2p_pipe, prompts, controller, latent=z_T, null_prompts=null_embeddings, run_baseline=False,
                               generator=generator, T=T, w=guidance_scale, fp=f"./results/{project_name}/p2p_reconstruct.png")
    pil_image = show_cross_attention(p2p_pipe, prompts, controller, res=16, from_where=("up", "down"),fp=f"./results/{project_name}/p2p_attention.png")
    buf = io.BytesIO()
    pil_image.save(buf, format='JPEG')
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


@app.route('/generate', methods=['POST', 'OPTIONS'])
def generate():
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

    try:
        yaml_file = request.form.get('config')
        print(yaml_file)
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            T = config["steps"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            T = int(request.form.get("steps"))
            print(project_name, source_prompt, T)
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

    pipe.scheduler.set_timesteps(T)
    with torch.inference_mode(), torch.autocast("cuda"):
        im = pipe(prompt=source_prompt, generator=generator)
        im[0][0].save(f"./results/{project_name}/no_guidance.png")

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

    try:
        yaml_file = request.form.get('config')
        print(yaml_file)
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            guidance_scale = config["guidance_scale"]
            T = config["steps"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            T = int(request.form.get("steps"))
            guidance_scale = float(request.form.get("guidance_scale"))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
    except Exception as e:
        return e, 400
    pipe.scheduler.set_timesteps(T)
    with torch.inference_mode(), torch.autocast("cuda"):
        z_T = init_trajectory[-1].to("cuda")
        im = pipe(prompt=source_prompt, latents=z_T, generator=generator, guidance_scale=guidance_scale)
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


@app.route('/null_recon', methods=['POST', 'OPTIONS'])
def null_recon():
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

    try:
        yaml_file = request.form.get('config')
        print(yaml_file)
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            guidance_scale = config["guidance_scale"]
            T = config["steps"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            T = int(request.form.get("steps"))
            guidance_scale = float(request.form.get("guidance_scale"))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)

    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")
        z_T = init_trajectory[-1]
    except Exception as e:
        return e, 400
    recon_img = reconstruct(pipe, z_T, source_prompt, null_embeddings, guidance_scale=guidance_scale, T=T)
    buf = io.BytesIO()
    print(recon_img.shape)
    im = Image.fromarray(np.uint8(recon_img[0] * 255)).convert('RGB')
    im.save(f"./results/{project_name}/null_recon.png", format="PNG")
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

    try:
        yaml_file = request.form.get('config')
        print(yaml_file)
        with open(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            project_name = config['project_name']
            source_prompt = config["prompt"]
            num_guidance = config["num_guidance"]
            T = config["steps"]
            guidance_scale = config["guidance_scale"]
            min_guidance = config["min_guidance"]
            max_guidance = config["max_guidance"]

    except Exception as e:
        print(e)
        try:
            project_name = request.form.get('project_name')
            source_prompt = request.form.get("prompt")
            guidance_scale = float(request.form.get("guidance_scale"))
            T = int(request.form.get("steps"))
            num_guidance = int(request.form.get("num_guidance"))
            min_guidance = float(request.form.get("min_guidance"))
            max_guidance = float(request.form.get("max_guidance"))
        except Exception as e:
            print(e)
            return Response(f"ERROR: {e}", status=500)
    try:
        init_trajectory = torch.load(f"./results/{project_name}/init_trajectory.pt")
        null_embeddings = torch.load(f"./results/{project_name}/nulls.pt")
        z_T = init_trajectory[-1]
    except Exception as e:
        return e, 400

    recon_img = reconstruct(pipe, z_T, source_prompt, null_embeddings, guidance_scale=guidance_scale, T=T)
    edit_imgs = []
    for scale in np.linspace(min_guidance, max_guidance, num_guidance):
        edit_img = reconstruct(pipe, z_T, source_prompt, null_embeddings, guidance_scale=scale, T=T)
        edit_imgs.append(edit_img)

    fig, ax = plt.subplots(1, num_guidance + 1, figsize=(10 * (num_guidance + 1), 10))
    ax[0].imshow(recon_img[0])
    ax[0].set_title("Reconstructed", fontdict={'fontsize': 40})
    ax[0].axis('off')

    for i, scale in enumerate(np.linspace(min_guidance, max_guidance, num_guidance)):
        ax[i + 1].imshow(edit_imgs[i][0])
        ax[i + 1].set_title("%.2f" % scale, fontdict={'fontsize': 40})
        ax[i + 1].axis('off')
    plt.xlabel(source_prompt)

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
    app.run(host='0.0.0.0', port=8080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080, debug=False, threaded=True)
