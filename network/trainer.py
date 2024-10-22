import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network.IPD_Net import IPD_Net
import os
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
from torch.utils.tensorboard import SummaryWriter

class trainer():
    def __init__(self, config, TEST=False):
        """ 如果是测试而不是训练或者验证, 传入TEST=false """

        self.config = config

        # 定义模型
        if config['network'] == "IPD_Net": # 已经确认是ResNet pretrained:ImageNet1k
            if config['pretrained_path'] != "": # 如果加载的是自己训练的模型
                self.model = IPD_Net(config)
                self.load_ckpt(config['pretrained_path'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            else:
                self.model = IPD_Net(config)
                print(f'[+]网络预训练模型加载成功')
        elif config['network'] == "efficientnet_b4.ra2_in1k": # Dataset: ImageNet-1k, Image size: train = 320 x 320
            if config['pretrained_path'] != "": # 如果加载的是自己训练的模型
                self.model = timm.create_model('efficientnet_b4.ra2_in1k', pretrained=False, num_classes=2)
                self.load_ckpt(config['pretrained_path'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            else:
                self.model = timm.create_model('efficientnet_b4.ra2_in1k', pretrained=True, num_classes=2)
                print(f'[+]网络预训练模型加载成功')
        elif config['network'] == "SRM->efficientnet_b4.ra2_in1k":
            from network.SRM_Net import SRM_Net
            if config['pretrained_path'] != "": # 如果加载的是自己训练的模型
                self.model = SRM_Net(name="efficientnet_b4.ra2_in1k", pretrained=False)
                self.load_ckpt(config['pretrained_path'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            else:
                self.model = SRM_Net(name="efficientnet_b4.ra2_in1k", pretrained=True)
                print(f'[+]网络预训练模型加载成功')
        elif config['network'] == "swin_base_patch4_window7_224_ms_in1k":
            if config['pretrained_path'] != "": # 如果加载的是自己训练的模型
                self.model = timm.create_model('swin_base_patch4_window7_224.ms_in1k', pretrained=False, num_classes=2)
                self.load_ckpt(config['pretrained_path'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            else:
                self.model = timm.create_model('swin_base_patch4_window7_224.ms_in1k', pretrained=True, num_classes=2)
                print(f'[+]网络预训练模型加载成功')
        elif config['network'] == "vit_base_patch16_224": # 这个tm是ImgaeNet21k...
            if config['pretrained_path'] != "": # 如果加载的是自己训练的模型
                self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
                self.load_ckpt(config['pretrained_path'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            else:
                self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
                print(f'[+]网络预训练模型加载成功')
        elif config['network'] == "vit_base_patch16_224_dino": # 这个是ImageNet1k
            if config['pretrained_path'] != "": # 如果加载的是自己训练的模型
                self.model = timm.create_model('vit_base_patch16_224.dino', pretrained=False, num_classes=2)
                self.load_ckpt(config['pretrained_path'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            else:
                self.model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=2)
                print(f'[+]网络预训练模型加载成功')
        elif config['network'] == "vit_base_patch16_224_dino_as_pretrained_vit_base_patch16_512_as_finetine":
            @register_model # 注册模型
            def vit_base_patch16_512(pretrained: bool = False, **kwargs):
                model_args = dict(img_size=512)
                model = _create_vision_transformer('vit_base_patch16_224.dino', pretrained=pretrained, **dict(model_args, **kwargs))
                return model
            if config['pretrained_path'] != "" and not config['flag']: # 这是加载老模型做微调
                self.model = timm.create_model('vit_base_patch16_512')
                cfg = self.model.default_cfg
                cfg['file'] = './tmp_file.pt'
                torch.save(torch.load(config['pretrained_path'])['model_state_dict'], cfg['file']) # 加载是224的权重
                self.model = timm.create_model('vit_base_patch16_512', pretrained=True, pretrained_cfg=cfg, num_classes=2)
                os.remove(cfg['file'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            elif config['pretrained_path'] != "" and config['flag']: # 这是加载模型做测试
                self.model = timm.create_model('vit_base_patch16_512')
                cfg = self.model.default_cfg
                cfg['file'] = './tmp_file.pt'
                torch.save(torch.load(config['placeholder'])['model_state_dict'], cfg['file']) # 加载是随便弄的224的权重,用来逼出他使用正确的模型,其实感觉随便一个权重都行,512的也行...
                self.model = timm.create_model('vit_base_patch16_512', pretrained=True, pretrained_cfg=cfg, num_classes=2)
                os.remove(cfg['file'])
                self.load_ckpt(config['pretrained_path'])
            else:
                self.model = timm.create_model('vit_base_patch16_512', pretrained=True, num_classes=2)
                print(f'[+]网络预训练模型加载成功')
        elif config['network'] == "tf_efficientnet_b3.ns_jft_in1k": # 这个是ImageNet1k
            if config['pretrained_path'] != "": # 如果加载的是自己训练的模型
                self.model = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', pretrained=False, num_classes=2)
                self.load_ckpt(config['pretrained_path'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            else:
                self.model = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', pretrained=True, num_classes=2)
                print(f'[+]网络预训练模型加载成功')
        elif config['network'] == "vit_base_patch16_224_as_pretrained_vit_base_patch16_512_as_finetine":
            @register_model # 注册模型
            def vit_base_patch16_512(pretrained: bool = False, **kwargs):
                model_args = dict(img_size=512)
                model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
                return model
            if config['pretrained_path'] != "" and not config['flag']: # 这是加载老模型然后微调
                self.model = timm.create_model('vit_base_patch16_512')
                cfg = self.model.default_cfg
                cfg['file'] = './tmp_file.pt'
                torch.save(torch.load(config['pretrained_path'])['model_state_dict'], cfg['file'])
                self.model = timm.create_model('vit_base_patch16_512', pretrained=True, pretrained_cfg=cfg, num_classes=2)
                os.remove(cfg['file'])
                print(f"[+]预训练模型[{config['pretrained_path']}]已加载成功")
            elif config['pretrained_path'] != "" and config['flag']: # 这是加载模型测试
                self.model = timm.create_model('vit_base_patch16_512')
                cfg = self.model.default_cfg
                cfg['file'] = './tmp_file.pt'
                torch.save(torch.load(config['placeholder'])['model_state_dict'], cfg['file'])
                self.model = timm.create_model('vit_base_patch16_512', pretrained=True, pretrained_cfg=cfg, num_classes=2)
                os.remove(cfg['file'])
                self.load_ckpt(config['pretrained_path'])
            else:
                self.model = timm.create_model('vit_base_patch16_512', pretrained=True, num_classes=2)
                print(f'[+]网络预训练模型加载成功')
        else:
            raise ValueError("[-]选定的神经网络结构未提供")
        print(f"[+]选定的神经网络结构为{config['network']}")


        # GPU设置
        self.device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu") # 设置主GPU
        self.model.to(self.device) # 设置主GPU


        if not TEST:
            self.ckpt_dir = os.path.join(self.config['save_ckpt_path'], self.config['name']) # 模型保存的文件夹
            self.writer = SummaryWriter(self.ckpt_dir) # 第一个参数指明 writer 把summary内容写在哪个目录下

            # 定义损失函数
            self.criterion = nn.CrossEntropyLoss()

            # 定义优化器
            optimizer_type = config['optimizer']['type']
            optimizer_args = config['optimizer'][optimizer_type]
            if config['optimizer']['type'] == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=optimizer_args['lr'],
                                    betas=(optimizer_args['beta1'],
                                            optimizer_args['beta2']),
                                    eps=optimizer_args['eps'],
                                    weight_decay=optimizer_args['weight_decay'],
                                    amsgrad=optimizer_args['amsgrad'])
            elif config['optimizer']['type'] == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(),
                                    lr=optimizer_args['lr'],
                                    momentum=optimizer_args['momentum'],
                                    weight_decay=optimizer_args['weight_decay'])
            else:
                raise ValueError("[-]选定的优化器未提供")

            # 定义学习率下降策略
            scheduler_type = config['scheduler']['type']
            scheduler_args = config['scheduler'][scheduler_type]
            if scheduler_type == 'CosineAnnealingLR':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_args['T_max'],
                    eta_min=scheduler_args['eta_min'],
                    last_epoch=scheduler_args['last_epoch'],
                    verbose=scheduler_args['verbose'])
            else:
                raise ValueError("[-]选定的学习率下载策略未提供")

            # 记录最佳ACC,一个epoch更新一次
            self.best_acc = 0.
            self.best_auc = 0.

            # 记录每个epoch结束后的测试的所有结果
            self.gt_list = None
            self.pre_list = None

    def scheduler_step(self):
        self.scheduler.step()

    def load_ckpt(self, ckpt_path):
        # 先加载到cpu上
        load_ = torch.load(ckpt_path, map_location=f"cuda:{self.config['gpu']}")
        self.model.load_state_dict(load_['model_state_dict'])
        print(f"[+]ACC: {load_['acc']}")
        print(f"[+]AUC: {load_['auc']}")

    def set_mode(self, mode):
        if mode == 'train':
            print("[+]训练模式启动")
            self.model.train()
        else:
            print("[+]验证模式启动")
            self.model.eval()

    def set_input(self, data_dict: dict):
        self.input = data_dict['image'].to(self.device)
        self.label = data_dict['label'].to(self.device)

    def forward(self):
        self.output = self.model(self.input)
        return self.output

    def get_loss(self):
        """ 获取一个batch的loss """
        return self.criterion(self.output, self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad() # 梯度清空
        self.loss.backward() # 反向传播
        """ for name, parms in self.model.named_parameters():
            print(f'-->name: {name}, -->grad_requires: {parms.requires_grad}, -->grad_value: {parms.grad}') """
        self.optimizer.step() # 梯度下降

    def test_reset(self):
        """ 每次验证开始前要重置 """
        self.gt_list = []
        self.pre_list = []

    def test_forward(self):
        self.forward()
        self.gt_list.extend(self.label[:, 1].cpu().numpy())
        self.output = F.softmax(self.output, dim=1) # 做一次softmax,因为是二分类
        self.pre_list.extend(self.output[:, 1].cpu().numpy())
        pass

    def test_finish(self):
        self.gt_list, self.pre_list = np.array(self.gt_list), np.array(self.pre_list)
        auc = roc_auc_score(self.gt_list, self.pre_list)
        acc = accuracy_score(self.gt_list, self.pre_list > 0.5)
        return auc,acc

    def save_ckpt(self, acc, auc, epoch):
        """ 模型保存 """
        self.writer.add_scalar("ACC", acc, epoch)
        self.writer.add_scalar("AUC", auc, epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        if acc > self.best_acc:
            self.best_acc = acc
        if auc > self.best_auc:
            self.best_auc = auc
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            ckpt_path = os.path.join(self.ckpt_dir, f'ckpt_best.pth')
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'acc': acc,
                'auc': auc,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"[+]最好的权重保存在[{ckpt_path}]")



if __name__ == "__main__":
    import yaml
    # 加载配置文件
    with open('training/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
