import gc

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from matplotlib import pyplot as plt

SD_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                               scheduler=DDIMScheduler(beta_start=0.00085,
                                                       beta_schedule="scaled_linear",
                                                       beta_end=0.012),
                                                  revision="fp16", torch_dtype=torch.float16,
                                               use_auth_token="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").to("cuda")