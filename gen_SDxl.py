from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
import os
import uuid  # 用于生成随机的唯一ID
import random
from tqdm import tqdm

########################################
gpu = 'cuda:3'
gen_num = 10000 # 要生成多少张图片
output_dir = "SDvxl"  # 要输出的文件夹
input_dir = "data/GMDD/phase1/trainset"  # 作为图生图的图片
########################################

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True)
pipe = pipe.to(gpu)


image_files = [f for f in os.listdir(input_dir)]

for i in range(gen_num):
    selected_image = random.choice(image_files)
    selected_image_path = os.path.join(input_dir, selected_image)

    prompt = "a human face"

    init_image = load_image(selected_image_path).convert("RGB")

    image = pipe(prompt, image=init_image).images[0]

    # 生成一个随机的UUID作为文件名
    random_filename = str(uuid.uuid4()) + '.png'
    image.save(os.path.join(output_dir, random_filename))
