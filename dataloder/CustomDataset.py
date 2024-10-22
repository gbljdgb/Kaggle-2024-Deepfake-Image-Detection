from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
from random import random, choice, randint
import cv2
from io import BytesIO
from scipy.ndimage.filters import gaussian_filter

def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def data_augment(img, config):
    img = np.array(img)

    if random() < config['blur_prob']:
        sig = sample_continuous(config['blur_sig'])
        gaussian_blur(img, sig)

    if random() < config['jpeg_prob']:
        method = sample_discrete(config['jpeg_method'])
        qual = randint(*config['jpeg_qual'])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)

def create_transforms(config, mode):
    if mode == 'train':
        isTrain = 1
    else:
        isTrain = 0

    if config['resize_or_crop'] == 'resize':
        size_func = transforms.Resize((config['input_shape'], config['input_shape']))
    elif config['resize_or_crop'] == 'crop':
        if isTrain:
            size_func = transforms.RandomCrop((config['input_shape'], config['input_shape']))
        else:
            size_func = transforms.CenterCrop((config['input_shape'], config['input_shape']))

    if isTrain: # 如果在训练模式下
        flip_func = transforms.RandomHorizontalFlip(p=config['flip_prob']) # 图像翻转
        rotate_func = transforms.RandomApply([transforms.RandomRotation(degrees=config['rotate_limit'])],p=config['rotate_prob']) # 图像旋转
        aug_func = transforms.Lambda(lambda img: data_augment(img, config)) # 图像压缩+高斯模糊
        brightness_func = transforms.RandomApply([transforms.ColorJitter(brightness=config['brightness_limit'], contrast=config['contrast_limit'])], p=config['brightness_prob'])
    else:
        flip_func = transforms.Lambda(lambda img: img)
        rotate_func = transforms.Lambda(lambda img: img)
        aug_func = transforms.Lambda(lambda img: img)
        brightness_func = transforms.Lambda(lambda img: img)

    return transforms.Compose([
        aug_func,
        flip_func,
        rotate_func,
        brightness_func,
        size_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# 定义自定义数据集类
class CustomDataset(Dataset):

    def __init__(self, config, mode):
        self.config = config # 配置
        self.mode = mode # 模式
        self.data = None
        self.length = None # 读取的数据集的长度

        if mode == "train":
            self.data = pd.read_csv(config['data']['train_label'])
            print(self.data['target'].value_counts()) # 打印训练集的0和1的个数

            # 平衡训练集中0和1的个数
            class_0 = self.data[self.data['target'] == 0]
            class_1 = self.data[self.data['target'] == 1]
            # 计算两类数据的数量差异
            difference = len(class_1) - len(class_0)
            # 复制类别较少的数据
            class_0_upsampled = class_0.sample(difference, replace=True)
            # 合并原始数据和复制的数据
            balanced_data = pd.concat([self.data, class_0_upsampled])
            # 打乱数据
            self.data = balanced_data.sample(frac=1).reset_index(drop=True)
            print(self.data['target'].value_counts())  # 打印训练集的0和1的个数

            self.length = len(self.data)
            self.transform = create_transforms(config, mode=mode)

        elif mode == 'val':
            self.data = pd.read_csv(config['data']['val_label'])
            print(self.data['target'].value_counts()) # 打印验证集的0和1的个数

            self.length = len(self.data)
            self.transform = create_transforms(config, mode=mode)

        elif mode == 'test': # 如果是测试模式
            self.data = pd.read_csv(config['data']['test_label'])

            self.length = len(self.data)
            self.transform = create_transforms(config, mode=mode)

        else:
            raise RuntimeError('数据集读取方式定义错误')

        if config['select_test'] != -1:
            self.length = config['select_test'] # 用来测试代码正确性,设置为100,则代表只选取100张图片

        if config['diFF_prob'] != 0.:
            print('[+]选择启用扩散模型数据集')
            folder_path = config['diff_path']
            self.diFF_file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path = os.path.join(self.config['data'][f'{self.mode}_dir_path'],self.data.iloc[idx]['img_name'])
            label = int(self.data.iloc[idx]['target'])

            if random() < self.config['diFF_prob'] and label == 1:
                img_path = choice(self.diFF_file_paths)

            sample = Image.open(img_path).convert('RGB')
            sample = self.transform(sample)

            one_hot = torch.zeros(2)
            one_hot[label] = 1.

            return sample, one_hot

        elif self.mode == 'val':
            img_path = os.path.join(self.config['data'][f'{self.mode}_dir_path'],self.data.iloc[idx]['img_name'])
            label = int(self.data.iloc[idx]['target'])

            sample = Image.open(img_path).convert('RGB')
            sample = self.transform(sample)

            one_hot = torch.zeros(2)
            one_hot[label] = 1.

            return sample, one_hot

        elif self.mode == 'test':
            img_path = os.path.join(self.config['data'][f'{self.mode}_dir_path'],self.data.iloc[idx]['img_name'])
            img_name = self.data.iloc[idx]['img_name'] # 这是写在CSV里面的
            ascii_values = [ord(c) for c in img_name]
            img_name_tensor = torch.tensor(ascii_values) # 图像名称转tensor

            sample = Image.open(img_path).convert('RGB')
            sample = self.transform(sample)

            return sample, img_name_tensor

        else:
            raise RuntimeError('ERROR!')

    @staticmethod
    def collate_fn(batch):
        # Separate the image, label, landmark, and mask tensors
        images, labels = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        return data_dict
