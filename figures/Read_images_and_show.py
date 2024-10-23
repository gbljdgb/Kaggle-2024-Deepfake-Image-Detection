import os, time
from PIL import Image
from tqdm import tqdm

# 此文件夹下全是图片
folder_path = "data/GMDD/phase1/trainset/"
folder_path = "data/GMDD/phase1/valset/"
# 要写入的图像的位置
output_img = "figures/output.png"

# 用于存储不同的图像尺寸
image_sizes = set()

# List the images in the folder
images = os.listdir(folder_path)

# Display the first image as an example
for idx in tqdm(range(len(images))):
    image_path = os.path.join(folder_path, images[idx])
    img = Image.open(image_path)
    image_sizes.add(img.size)
    if idx % 50000 == 0:
        print(image_sizes)
    # img.save(output_img)

# 输出不同图像尺寸的数量
print(f"共有 {len(image_sizes)} 种不同的图像尺寸")

# 如果需要，可以打印所有不同的图像尺寸
print("不同的图像尺寸如下：")
for size in image_sizes:
    print(size)
