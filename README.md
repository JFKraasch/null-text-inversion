# null-text-inversion
Unofficial implementation of paper NULL-text Inversion for Editing Real Images using Guided Diffusion Models ( https://arxiv.org/abs/2211.09794 )

In the paper new method for Image inversion using Diffusion models are proposed. Read to get more details.

# Webaverse implementation
Create config as as seen in configs/room.yaml and call app.py (port 5000)\
The resuling images are stored at './results/{project name}/'

post to addresses to call functions:

1. /DDIM -> extracts latents
2. /inversion -> performs null-text inversion
3. /edit -> uses edited prompt to edit image

(Utils for testing)\
/ddim_recon -> reconstruct image without null inversion\
/guidance -> test results with different guidance values

call /edit with edited prompt == prompt to reconstruct image

## Installation
```
conda create --name textinv python=3.8 -y
conda activate textinv
conda install pytorch torchvision cudatoolkit pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt
```
