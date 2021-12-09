'''
작성자 김지성, 홍지연
최종 수정일 2021-12-09
'''

import sys
import os
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

sys.path.append(os.path.join(os.getcwd(), './lama'))
from lama.saicinpainting.training.trainers import load_checkpoint
from lama.saicinpainting.evaluation.data import pad_img_to_modulo
from lama.saicinpainting.evaluation.utils import move_to_device

class LaMa:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)

        with open(r'big-lama/config.yaml', 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'

            self.model = load_checkpoint(train_config, 'big-lama/models/best.ckpt', strict=False, map_location='cpu')
            self.model.freeze()
            self.model.to(self.device)

    def inference(self, image, mask):
        image = np.transpose(image, (2, 0, 1))
        image = image.astype('float32') / 255

        result = dict(image=image, mask=mask[None, ...])
        result['image'] = pad_img_to_modulo(result['image'], 8)
        result['mask'] = pad_img_to_modulo(result['mask'], 8)

        batch = move_to_device(default_collate([result]), self.device)

        batch['mask'] = (batch['mask'] > 0) * 1
        batch = self.model(batch)
        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

        return np.clip(cur_res * 255, 0, 255).astype('uint8')