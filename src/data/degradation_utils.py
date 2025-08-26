import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from PIL import Image
import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _add_poisson_noise(self, clean_patch, scale=1.0):
        """
        添加泊松噪声到图像块

        参数:
            clean_patch: 干净的图像块(值范围应在0-255之间)
            scale: 控制噪声强度的缩放因子(默认1.0)

        返回:
            noisy_patch: 添加噪声后的图像块
            clean_patch: 原始干净图像块(与输入一致)
        """
        # 将图像归一化到0-1范围
        clean_normalized = clean_patch.astype(np.float32) / 255.0

        # 生成泊松噪声
        noisy_normalized = np.random.poisson(clean_normalized * scale) / np.clip(scale, 1e-10, None)

        # 将图像重新缩放到0-255范围并裁剪
        noisy_patch = np.clip(noisy_normalized * 255, 0, 255).astype(np.uint8)

        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)
        elif degrade_type == 3:
            # poisson noise
            degraded_patch, clean_patch = self._add_poisson_noise(clean_patch, scale=1000.0)
        else:
            raise NotImplementedError(f"Degradation type {degrade_type} not defined.")

        return degraded_patch, clean_patch

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            deg_dict = {
                'denoise_15': 0,
                'denoise_25': 1,
                'denoise_50': 2,
                'poisson_1': 3,
            }
            degrade_type = deg_dict[degrade_type]

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1
