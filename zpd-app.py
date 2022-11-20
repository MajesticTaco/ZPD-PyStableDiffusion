import os
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

#Šeit ievada tekstu
prompt = 'dog'

print(f"Characters in prompt: {len(prompt)}, limit 200")

pipe = StableDiffusionPipeline.from_pretrained(SDV5_MODEL_PATH)
pipe = pipe.to('cuda')

with autocast('cuda'):
    image = pipe(prompt).images[0]

image_path = savedif(os.path(SAVE_PATH, (prompt[:25] + '...') if len(prompt) > 25 else prompt) + '.png')

image.save(image_path)