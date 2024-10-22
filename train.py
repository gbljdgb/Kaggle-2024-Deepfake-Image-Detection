import yaml
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import datetime

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
    with open('options/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        if not os.path.exists(os.path.join(config['save_ckpt_path'],config['name'])):
            os.mkdir(os.path.join(config['save_ckpt_path'], config['name']))
    with open(os.path.join(config['save_ckpt_path'], config['name'], 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # 获取时间戳
    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[+]起始时间:{dt}")

    # 设置随机数种子
    setup_seed(config['manualSeed'])

    # 创建自定义数据集
    train_dataset = CustomDataset(config, mode='train')
    val_dataset = CustomDataset(config, mode='val')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_batchSize'],
                              num_workers=config['workers'],
                              collate_fn=train_dataset.collate_fn,
                              shuffle=True)
    len_train_dataloader = len(train_loader)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['val_batchSize'],
                            num_workers=config['workers'],
                            collate_fn=val_dataset.collate_fn,
                            shuffle=False)

    # 创建模型实例
    TRAINER = trainer(config)

    # 训练模型
    for epoch in range(config['nEpochs']):

        TRAINER.set_mode(mode='train')

        for index, data_dict in enumerate(tqdm(train_loader)):

            TRAINER.set_input(data_dict)
            TRAINER.optimize_parameters()

            if len(train_loader)//config['printFreq'] == 0:
                config['printFreq'] = 1
            if index % (len(train_loader)//config['printFreq']) == 0:
                print(f"[+]Batch Loss: {TRAINER.loss:.8f}")

        TRAINER.set_mode(mode='eval')

        with torch.no_grad():
            TRAINER.test_reset() # 重置指标
            for data_dict in tqdm(val_loader):
                TRAINER.set_input(data_dict)
                TRAINER.test_forward()
            auc, acc = TRAINER.test_finish()

        TRAINER.save_ckpt(acc, auc, epoch)
        print(f"[+]Epoch[{epoch}], ACC:[{acc}], AUC:[{auc}]")

        TRAINER.scheduler_step() # 学习率调整

    TRAINER.writer.close()
    print(f"[+]tensorboard销毁成功")
    print("[+]Train Finish!")
