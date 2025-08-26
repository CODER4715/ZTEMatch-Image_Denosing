import os
import numpy as np
import torch
from net.moce_ir import MoCEIR
from options import train_options
import yaml
from ptflops import get_model_complexity_info


def main(opt):
    # np.random.seed(42)
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    model = MoCEIR(
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
        complexity_scale=opt.complexity_scale,
        )

    checkpoint = torch.load(os.path.join(opt.ckpt_dir, opt.checkpoint_id, "last.ckpt"))
    # 只加载主模型权重
    model_state_dict = {k.replace('net.', ''): v for k, v in checkpoint['state_dict'].items()
                        if k.startswith('net.')}

    # 加载过滤后的权重
    model.load_state_dict(model_state_dict, strict=True)

    model.cuda()
    model.eval()
    macs, params = [], []
    for i in range(100):
        mac, param = get_model_complexity_info(model, (3, 720, 1280), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        # print(mac)
        macs.append(float(mac.split(' ')[0]))
        params.append(float(param.split(' ')[0]))

    print('Mean MACs: ', np.mean(macs))
    print('Mean Params: ', np.mean(params))




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
    train_opt.model = 'MoCE_IR_30G'
    hparams_path = os.path.join('checkpoints', train_opt.checkpoint_id, 'hparams.yaml')
    update_model_params(train_opt, hparams_path)
    main(train_opt)