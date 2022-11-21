import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

SDV5_MODEL_PATH = os.path.expanduser("~/stable-diffusion-v1-5")
SAVE_PATH = os.getcwd()

def savedif(path):
    filename, extension = os.path.splitext(path)
    counter = 1 

    while os.path.exists(path):
        path = filename + ' (' + str(counter) +')'+ extension
        counter += 1

    return path


prompt = '' # Šeit ievada tekstu

print(f"Rakstzīmes ievadē: {len(prompt)}, ieteicamais limits 200")

pipe = StableDiffusionPipeline.from_pretrained(SDV5_MODEL_PATH, revision="fp16", torch_dtype=torch.float16) # Float16 var noņemt ja hostam ir vairāk kā 10240MiB VRAM, šis variants izmanto +/- 6516MiB

pipe = pipe.to('cuda')

with autocast('cuda'):
    image = pipe(prompt).images[0]

prompt_u = prompt.replace(' ' , '_')

image_path = savedif(os.path.join(SAVE_PATH, (prompt_u[:25] + '...') if len(prompt_u) > 25 else prompt_u) + '.png')

image.save(image_path)

print('Attēls saglabāts: ' + image_path)
