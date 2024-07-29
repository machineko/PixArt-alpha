import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
from PIL import Image
from pathlib import Path
import os
import numpy as np
os.environ['HF_HOME'] = '/raid/models'
device = torch.device("cuda:1")

def pred(img_name: str):

    base_path = "data/_Benchmark_photos/"
    all_masks = [i for i in os.listdir(f"{base_path}/Masks") if i.__contains__(img_name[:-4])]
    image_path = Path(base_path, img_name).__str__()
    caption = captions[img_name[:-4]]
    for mask in all_masks:
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(Path(base_path, "Masks", mask).__str__()).convert("RGB")
        generator = torch.Generator(device="cuda:1").manual_seed(0)

        image = pipe(
            prompt="",
            image=image,
            mask_image=mask_image,
            guidance_scale=8.0,
            num_inference_steps=30,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=generator,
        ).images[0]
        image.save(f"outdreamshaper/d8_noprompt_{mask}")



pipe = StableDiffusion3InpaintPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

pipe.to(device)

all_files = [i for i in os.listdir("data/_Benchmark_photos") if "jpg" in i or ".png" in i]

for f in all_files:
    pred(f)