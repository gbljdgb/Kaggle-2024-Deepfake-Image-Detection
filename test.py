import yaml
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import datetime
import csv
import torch.nn.functional as F

from dataloder.CustomDataset import CustomDataset
from network.trainer import trainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # 加载配置文件
    with open('options/test_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        if not os.path.exists(os.path.join(config['save_output_path'],config['name'])):
            os.mkdir(os.path.join(config['save_output_path'], config['name']))
    with open(os.path.join(config['save_output_path'], config['name'], 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # 获取时间戳
    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[+]起始时间:{dt}")

    # 设置随机数种子
    setup_seed(config['manualSeed'])

    # 创建自定义数据集
    test_dataset = CustomDataset(config, mode='test')

    # 创建数据加载器
    test_loader = DataLoader(test_dataset,
                             batch_size=config['test_batchSize'],
                             num_workers=config['workers'],
                             collate_fn=test_dataset.collate_fn,
                             shuffle=False)

    # 创建模型实例
    TRAINER = trainer(config, TEST=True)

    # 创建一个字典来存储图像文件名和预测类别索引
    prediction_results = []

    # 评估模型
    TRAINER.set_mode(mode='eval')

    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            TRAINER.set_input(data_dict)
            TRAINER.forward()
            TRAINER.output = F.softmax(TRAINER.output, dim=1)

            # 为每个图像记录预测结果
            for i, prediction in zip(TRAINER.label, TRAINER.output):
                img_name = ''.join([chr(num.item()) for num in i])
                pred_prob = prediction[1].cpu().item()
                # print(pred_prob)
                prediction_results.append([img_name, pred_prob])
            # exit(0)

    # 指定输出的 JSON 文件路径
    output_csv_file = os.path.join(config['save_output_path'], config['name'], 'output.csv')

    # 将结果写入 CSV 文件
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['img_name', 'y_pred'])
        # 写入每一行数据
        writer.writerows(prediction_results)

    print(f"Results saved to {output_csv_file}")
