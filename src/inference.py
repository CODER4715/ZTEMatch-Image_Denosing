import os
import pathlib
import argparse
import numpy as np

from tqdm import tqdm
from typing import List
from skimage.util import img_as_ubyte

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from net.moce_ir import MoCEIR
from options import train_options
from utils.test_utils import save_img
from data.dataset_utils import Inference
import cv2

import yaml
from argparse import Namespace


####################################################################################################
## PL Test Model
class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.net = MoCEIR(
            dim=opt.dim,
            num_blocks=opt.num_blocks,
            num_dec_blocks=opt.num_dec_blocks,
            levels=len(opt.num_blocks),
            heads=opt.heads,
            num_refinement_blocks=opt.num_refinement_blocks,
            topk=opt.topk,
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity,
            depth_type=opt.depth_type,
            stage_depth=opt.stage_depth,
            rank_type=opt.rank_type,
            complexity_scale=opt.complexity_scale, )

    def forward(self, x):
        return self.net(x)


def run_inference(opts, net, dataset):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    if opts.save_results:
        pathlib.Path(os.path.join(os.getcwd(), f"{opts.output_path}/{opts.checkpoint_id}")).mkdir(parents=True,
                                                                                                  exist_ok=True)
    save_folder = os.path.join(os.getcwd(), f"{opts.output_path}/{opts.checkpoint_id}")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print("save to ", save_folder)
    with torch.no_grad():
        for (clean_name, degrad_patch, h, w) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()

            # Forward pass
            restored = net(degrad_patch)
            if isinstance(restored, List) and len(restored) == 2:
                restored, _ = restored

            # save output images
            restored = torch.clamp(restored, 0, 1)

            restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored = img_as_ubyte(restored)
            # resize to original size
            restored = cv2.resize(restored, (w.item(), h.item()), interpolation=cv2.INTER_CUBIC)
            # restored = cv2.GaussianBlur(restored, (3, 3), 0)
            restored = cv2.bilateralFilter(restored, d=7, sigmaColor=75, sigmaSpace=75)

            ext = os.path.splitext(clean_name[0])[-1]
            save_name = os.path.splitext(os.path.split(clean_name[0])[-1])[0] + ext
            save_img(
                os.path.join(f"{opts.output_path}/{opts.checkpoint_id}",
                             save_name),
                restored)


def main(opt):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PLTestModel(opt).to(device)
    checkpoint = torch.load(os.path.join(opt.ckpt_dir, opt.checkpoint_id, "last.ckpt"))
    # 打印所有state_dict的key
    # print("Checkpoint keys:")
    # for key in checkpoint['state_dict'].keys():
    #     print(key)

    # 只加载主模型权重
    model_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()
                        if k.startswith('net.')}

    # 加载过滤后的权重
    model.net.load_state_dict(model_state_dict, strict=False)
    model.eval()

    ind_opt = opt
    dataset = Inference(ind_opt)

    print("--------> Inference on", "testset.")
    print("\n")
    run_inference(opt, model, dataset)


def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string


def update_model_params(train_opt, yaml_path='hparams.yaml'):
    """只更新模型结构相关参数"""
    MODEL_KEYS = {
        'stage_depth', 'topk', 'num_blocks', 'num_dec_blocks',
        'num_exp_blocks', 'num_refinement_blocks', 'depth_type',
        'dim', 'heads', 'latent_dim', 'complexity_scale'
    }

    with open(yaml_path, 'r') as f:
        hparams = yaml.safe_load(f)

    # 只更新模型结构参数
    for k in MODEL_KEYS:
        if k in hparams:
            setattr(train_opt, k, hparams[k])

    return train_opt


if __name__ == '__main__':
    train_opt = train_options()
    hparams_path = os.path.join('checkpoints', train_opt.checkpoint_id, 'hparams.yaml')  # 替换为你的YAML文件路径
    # 3. 更新train_opt中的参数
    print(hparams_path)
    update_model_params(train_opt, hparams_path)
    main(train_opt)
