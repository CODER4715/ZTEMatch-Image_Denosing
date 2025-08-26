from typing import List
import os
import pathlib
import numpy as np
from tqdm import tqdm
from datetime import datetime

import random
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from net.moce_ir import MoCEIR

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import *
from utils.loss_utils import *

from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio as psnr
from lightning.pytorch import seed_everything
import signal
import yaml

class InterruptCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.original_handler = None

    def on_train_start(self, trainer, pl_module):
        def handle_interrupt(signum, frame):
            print("\n捕获到中断信号，正在保存模型...")
            trainer.save_checkpoint(os.path.join(self.save_path, "last.ckpt"))
            print(f"模型已保存到: {os.path.join(self.save_path, 'last.ckpt')}")
            if self.original_handler is not None:
                self.original_handler(signum, frame)
            exit(0)

        self.original_handler = signal.signal(signal.SIGINT, handle_interrupt)

    def on_train_end(self, trainer, pl_module):
        if self.original_handler is not None:
            signal.signal(signal.SIGINT, self.original_handler)

class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.validation_step_outputs = []
        self.save_hyperparameters(opt)
        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight

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

        if opt.loss_fn == "L1":
            self.loss_fn = nn.L1Loss()
            print("Using L1 loss")
        elif opt.loss_fn == "L2":
            self.loss_fn = nn.MSELoss()
            print("Using MSE Loss")
        else:
            raise ValueError(f"不支持的损失函数类型: {opt.loss_fn}")

        print(f"Pixel loss weight: {opt.pixel_loss_weight}")

        if opt.loss_type == "vgg":
            # Initialize VGG19 loss
            self.vgg_loss = VGG19Loss(feature_layer=35)  # Using relu5_4 features
            print("Using VGG19 perceptual loss")
            print(f"VGG loss weight: {opt.vgg_loss_weight}")

        elif opt.loss_type == "multi_vgg":
            # Initialize multi-layer VGG loss
            self.vgg_loss = MultiLayerVGGLoss(
                feature_layers=opt.vgg_layers ,
                layer_weights=opt.vgg_layer_weights if hasattr(opt, 'vgg_layer_weights') else None
            )
            print("Using multi-layer VGG perceptual loss")
            print(f"VGG loss weight: {opt.vgg_loss_weight}")

        if  opt.use_fft_loss:
            self.fft_loss = FFTLoss(loss_weight=self.opt.fft_loss_weight, fft_loss_type=self.opt.fft_loss_fn)
            print("Using FFT loss")
            print(f"FFT loss weight: {opt.fft_loss_weight}")
            print(f"FFT loss fn: {opt.fft_loss_fn}")

        if opt.ms_ssim_loss:
            self.ms_ssim_fn = MSSSIMLoss(
                levels=opt.ms_ssim_levels,
                window_size=opt.ms_ssim_window_size
            )
            print("Using MS-SSIM loss")
            print(f"MS-SSIM loss weight: {opt.ms_ssim_weight}")

        if opt.psnr_loss:
            self.psnr_fn = PSNRLoss()
            print("Using PSNR loss")
            print(f"PSNR loss weight: {opt.psnr_weight}")

        if opt.aux_l2_loss:
            self.aux_l2_loss = nn.MSELoss()
            print("Using Aux L2 loss")
            print(f"Aux L2 loss weight: {opt.aux_l2_weight}")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):

        # 确保VGG处于eval模式
        if hasattr(self, 'vgg_loss'):
            self.vgg_loss.eval()

        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id)

        loss = self.loss_fn(restored, clean_patch) * self.opt.pixel_loss_weight
        pixel_loss = loss

        balance_loss = self.net.total_loss  # balance_loss是专家的损失
        balance_loss = self.balance_loss_weight * balance_loss
        loss = loss + balance_loss
        self.log("Train_expert_loss", balance_loss, sync_dist=True)


        if self.opt.loss_type == "vgg" or self.opt.loss_type == "multi_vgg":
            vgg_loss = self.vgg_loss(restored, clean_patch)
            vgg_loss = self.opt.vgg_loss_weight * vgg_loss
            loss = loss + vgg_loss

            # Log individual layer losses if available
            if hasattr(self.vgg_loss, 'current_layer_losses'):
                for name, val in self.vgg_loss.current_layer_losses.items():
                    self.log(f"Train_{name}", val, sync_dist=True)
            self.log("Train_VGG_Loss", vgg_loss, sync_dist=True)

        if self.opt.use_fft_loss:
            fft_loss = self.fft_loss(restored, clean_patch)
            loss = loss + fft_loss
            self.log("Train_FFT_Loss", fft_loss, sync_dist=True)

        # 添加MS-SSIM损失
        if self.opt.ms_ssim_loss:
            ms_ssim_loss = self.ms_ssim_fn(restored, clean_patch)
            ms_ssim_loss = self.opt.ms_ssim_weight * ms_ssim_loss
            loss = loss + ms_ssim_loss
            self.log("Train_MS_SSIM_Loss", ms_ssim_loss, sync_dist=True)

        if self.opt.psnr_loss:
            psnr_loss = self.psnr_fn(restored, clean_patch)
            psnr_loss = self.opt.psnr_weight * psnr_loss
            loss = loss + psnr_loss
            self.log("Train_PSNR_Loss", psnr_loss, sync_dist=True)

        if self.opt.aux_l2_loss:
            aux_l2_loss = self.aux_l2_loss(restored, clean_patch)
            aux_l2_loss = self.opt.aux_l2_weight * aux_l2_loss
            loss = loss + aux_l2_loss
            self.log("Train_Aux_L2_Loss", aux_l2_loss, sync_dist=True)

        self.log("Train_Pixel({})_Loss".format(self.opt.loss_fn), pixel_loss, sync_dist=True)
        self.log("Train_Loss", loss, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, sync_dist=True)

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=int(0.1 * self.opt.epochs),
                                                  max_epochs=self.opt.epochs)

        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=1, max_epochs=self.opt.epochs)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):

        (clean_name, degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        pixel_loss = loss

        balance_loss = self.net.total_loss  # balance_loss是专家的损失
        balance_loss = self.balance_loss_weight * balance_loss
        loss = loss + balance_loss

        if self.opt.loss_type == "vgg" or self.opt.loss_type == "multi_vgg":
            vgg_loss = self.vgg_loss(restored, clean_patch)
            vgg_loss = self.opt.vgg_loss_weight * vgg_loss
            loss = loss + vgg_loss

            # Log individual layer losses if available
            if hasattr(self.vgg_loss, 'current_layer_losses'):
                for name, val in self.vgg_loss.current_layer_losses.items():
                    self.log(f"Train_{name}", val, sync_dist=True)

        if self.opt.use_fft_loss:
            fft_loss = self.fft_loss(restored, clean_patch)
            loss = loss + fft_loss

        if self.opt.ms_ssim_loss:
            ms_ssim_loss = self.ms_ssim_fn(restored, clean_patch)
            ms_ssim_loss = self.opt.ms_ssim_weight * ms_ssim_loss
            loss = loss + ms_ssim_loss

        if self.opt.psnr_loss:
            psnr_loss = self.psnr_fn(restored, clean_patch)
            psnr_loss = self.opt.psnr_weight * psnr_loss
            loss = loss + psnr_loss

        if self.opt.aux_l2_loss:
            aux_l2_loss = self.aux_l2_loss(restored, clean_patch)
            aux_l2_loss = self.opt.aux_l2_weight * aux_l2_loss
            loss = loss + aux_l2_loss

        # 计算 PSNR
        psnr_value = psnr(clean_patch.cpu().numpy(), restored.cpu().numpy(), data_range=1.0)

        self.validation_step_outputs.append({
            "loss": loss,
            "psnr": psnr_value,
            "pixel_loss": pixel_loss,
            "balance_loss": balance_loss,
            "vgg_loss": vgg_loss if hasattr(self, 'vgg_loss') else None,
            "fft_loss": fft_loss if hasattr(self, 'fft_loss') else None,
            "ms_ssim_loss": ms_ssim_loss if hasattr(self, 'ms_ssim_fn') else None,
            "psnr_loss": psnr_loss if hasattr(self, 'psnr_fn') else None,
            "aux_l2_loss": aux_l2_loss if hasattr(self, 'aux_l2_loss') else None,
            "clean_patch": clean_patch,
            "restored": restored
        })

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs

        # 1. 计算平均 loss 和 PSNR
        avg_loss = torch.tensor([x["loss"] for x in validation_step_outputs]).mean()
        avg_psnr = torch.tensor([x["psnr"] for x in validation_step_outputs]).mean()
        avg_balance_loss = torch.tensor([x["balance_loss"] for x in validation_step_outputs]).mean()
        avg_vgg_loss = torch.tensor(
            [x["vgg_loss"] for x in validation_step_outputs if x["vgg_loss"] is not None]).mean()
        avg_fft_loss = torch.tensor(
            [x["fft_loss"] for x in validation_step_outputs if x["fft_loss"] is not None]).mean()
        avg_ms_ssim_loss = torch.tensor(
            [x["ms_ssim_loss"] for x in validation_step_outputs if x["ms_ssim_loss"] is not None]).mean()
        avg_psnr_loss = torch.tensor(
            [x["psnr_loss"] for x in validation_step_outputs if x["psnr_loss"] is not None]).mean()
        avg_aux_l2_loss = torch.tensor(
            [x["aux_l2_loss"] for x in validation_step_outputs if x["aux_l2_loss"] is not None]).mean()
        avg_pixel_loss = torch.tensor([x["pixel_loss"] for x in validation_step_outputs]).mean()

        # 记录每个损失
        self.log("val_loss", avg_loss, sync_dist=True)
        self.log("val_psnr", avg_psnr, sync_dist=True)
        self.log("val_balance_loss", avg_balance_loss, sync_dist=True)
        self.log("val_pixel_loss", avg_pixel_loss, sync_dist=True)
        if hasattr(self, 'vgg_loss'):
            self.log("val_vgg_loss", avg_vgg_loss, sync_dist=True)
        if hasattr(self, 'fft_loss'):
            self.log("val_fft_loss", avg_fft_loss, sync_dist=True)
        if hasattr(self, 'ms_ssim_fn'):
            self.log("val_ms_ssim_loss", avg_ms_ssim_loss, sync_dist=True)
        if hasattr(self, 'psnr_fn'):
            self.log("val_psnr_loss", avg_psnr_loss, sync_dist=True)
        if hasattr(self, 'aux_l2_loss'):
            self.log("val_aux_l2_loss", avg_aux_l2_loss, sync_dist=True)

        # 2. 随机选择 3 个样本可视化
        num_samples = min(5, len(validation_step_outputs))
        indices = torch.randperm(len(validation_step_outputs))[:num_samples]

        # 3. 记录对比图像（clean和restored并排显示）
        if self.logger:
            for idx in indices:
                clean_img = validation_step_outputs[idx]["clean_patch"]  # 形状应该是 [C, H, W]
                restored_img = validation_step_outputs[idx]["restored"]

                # 确保图像是3D张量 [C, H, W]
                if len(clean_img.shape) == 4:  # 如果有batch维度 [1, C, H, W]
                    clean_img = clean_img.squeeze(0)
                    restored_img = restored_img.squeeze(0)

                # 将clean和restored图像水平拼接
                comparison = torch.cat([clean_img, restored_img], dim=-1)  # 结果形状 [C, H, W*2]

                # 记录到TensorBoard
                self.logger.experiment.add_image(
                    f"val_samples/comparison_{idx}",
                    comparison,
                    global_step=self.global_step,
                    dataformats='CHW'  # 明确指定格式为CHW
                )

        self.validation_step_outputs.clear()


def update_params(train_opt, yaml_path='hparams.yaml'):
    with open(yaml_path, 'r') as f:
        hparams = yaml.safe_load(f)
    for k in hparams:
            setattr(train_opt, k, hparams[k])

    return train_opt

def main(opt):
    # print("Options")
    # print(opt)
    seed_everything(42)
    if opt.exp_name is None:
        time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    else:
        time_stamp = opt.exp_name

    torch.set_float32_matmul_precision('high')

    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    if opt.wblogger:
        name = opt.model + "_" + time_stamp
        logger = WandbLogger(name=name, save_dir=log_dir, config=opt)

    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    if opt.fine_tune_from:
        if opt.loss_type == "vgg" or opt.loss_type == "multi_vgg":
            model = PLTrainModel.load_from_checkpoint(
                os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt)
        else:
            model = PLTrainModel.load_from_checkpoint(
                os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt, strict=False)
    else:
        model = PLTrainModel(opt)

    print(model)
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, every_n_epochs=4, save_top_k=-1, save_last=True)
    interrupt_callback = InterruptCallback(checkpoint_path)  # 新增中断回调


    trainset = H5_AIOTrainDataset_2(opt)

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True, drop_last=True,
                             num_workers=opt.num_workers)

    valset = Validation(opt)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    # Create trainer
    trainer = pl.Trainer(max_epochs=opt.epochs,
                         precision='32-true',
                         accelerator="gpu",
                         devices=opt.num_gpus,
                         # strategy="ddp_find_unused_parameters_true",#
                         logger=logger,
                         callbacks=[checkpoint_callback, interrupt_callback],
                         accumulate_grad_batches=opt.accum_grad,
                         deterministic=True,
                         # check_val_every_n_epoch=4,
                         val_check_interval=5000,
                         )

    # Optionally resume from a checkpoint
    if opt.resume_from:
        checkpoint_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
    else:
        checkpoint_path = None

    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=trainloader,
        val_dataloaders=valloader,
        ckpt_path=checkpoint_path  # Specify the checkpoint path to resume from
    )


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)
