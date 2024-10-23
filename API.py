import yaml
import torch
from flask import Flask, request, jsonify
import numpy as np
import random
import torch.nn.functional as F
from network.trainer import trainer
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# 加载配置文件
with open('options/test_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

setup_seed(config['manualSeed'])

# 创建模型实例
TRAINER = trainer(config, TEST=True)
TRAINER.set_mode(mode='eval')

transform = transforms.Compose([transforms.Resize((config['input_shape'], config['input_shape'])),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    # 这里假设有个方法来处理图片并转换为合适的输入格式
    data_dict = preprocess_image(image_file)

    with torch.no_grad():
        TRAINER.set_input(data_dict)
        TRAINER.forward()
        TRAINER.output = F.softmax(TRAINER.output, dim=1)

        pred_prob = TRAINER.output[0][1].cpu().item()  # 只取第一个预测
        return jsonify({'probability': pred_prob})


def preprocess_image(image_file):
    # 实现图片预处理的方法，返回一个合适的data_dict
    image = Image.open(image_file.stream).convert('RGB')
    sample = transform(image)
    data_dict = {}
    data_dict['image'] = sample.unsqueeze(0)
    data_dict['label'] = sample.unsqueeze(0)
    return data_dict

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10086)
