import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.models import create_model
from furnace.optim_factory import create_optimizer

from furnace.datasets import build_cae_pretraining_dataset
from furnace.engine_for_pretraining import train_one_epoch
from furnace.utils import NativeScalerWithGradNormCount as NativeScaler
import furnace.utils as utils


# 获取命令行参数
def get_args():
    parser = argparse.ArgumentParser('pre-training script', add_help=False)

    # 模型类型选择
    parser.add_argument('--model_type', default='cae', help='选择模型类型', choices=['cae', 'caev2'])
    parser.add_argument('--batch_size', default=64, type=int, help='批处理大小')
    parser.add_argument('--epochs', default=300, type=int, help='训练的总周期数')
    parser.add_argument('--save_ckpt_freq', default=50, type=int, help='保存检查点的频率')
    parser.add_argument("--discrete_vae_weight_path", type=str, help='离散VAE权重路径')
    parser.add_argument("--discrete_vae_type", type=str, default="dall-e",
                        help='[dall-e, vqgan_gumbel_f8_8192, clip, customized]')
    parser.add_argument('--dvae_num_layers', default=3, type=int, help='离散VAE的层数')

    # 模型参数
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL', help='要训练的模型名称')
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1适用于基础模型，1e-5适用于大型模型。设置为0禁用层级缩放")

    parser.add_argument('--input_size', default=224, type=int, help='输入图像的大小')
    parser.add_argument('--second_input_size', default=112, type=int, help='离散VAE的输入图像大小')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT', help='丢弃路径的比例（默认：0）')

    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='优化器（默认： "adamw"）')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='优化器的Epsilon（默认：1e-8）')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='优化器的Betas（默认：0.9, 0.98，使用优化器默认值）')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='裁剪梯度范数（默认：无裁剪）')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD动量（默认：0.9）')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减（默认：0.05）')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="最终的权重衰减值。使用余弦调度的权重衰减。")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='学习率（默认：5e-4）')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='热身学习率（默认：1e-6）')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='循环调度的最小学习率（默认：1e-5）')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='热身学习率的周期数')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='热身学习率的步骤数')

    # 数据增强参数
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='训练插值方法（随机、双线性、双三次，默认： "bicubic"）')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='离散VAE的插值方法（随机、双线性、双三次，默认： "lanczos"）')

    # 数据集参数
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='数据集路径')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true',
                        help='使用ImageNet的默认均值和标准差')

    parser.add_argument('--output_dir', default='', help='保存路径，如果为空则不保存')
    parser.add_argument('--log_dir', default=None, help='Tensorboard日志的保存路径')
    parser.add_argument('--device', default='cuda', help='用于训练/测试的设备')
    parser.add_argument('--seed', default=0, type=int, help='随机种子')
    parser.add_argument('--resume', default='', help='从检查点恢复')
    parser.add_argument('--auto_resume', action='store_true', help='自动恢复')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume', help='不自动恢复')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='开始的周期')
    parser.add_argument('--num_workers', default=10, type=int, help='用于数据加载的工作线程数')
    parser.add_argument('--pin_mem', action='store_true', help='在DataLoader中固定CPU内存，以便更高效地传输到GPU。')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int, help='分布式进程的数量')
    parser.add_argument('--local_rank', default=-1, type=int, help='本地排名')
    parser.add_argument('--dist_on_itp', action='store_true', help='在交互式模式下使用分布式训练')
    parser.add_argument('--dist_url', default='env://', help='用于设置分布式训练的URL')

    parser.add_argument('--exp_name', default='', type=str, help='用于保存检查点时的名称')
    parser.add_argument('--enable_multi_print', action='store_true', default=False, help='允许每个GPU打印信息')

    '''
    数据增强
    '''
    # 裁剪大小
    parser.add_argument('--crop_min_size', type=float, default=0.08, help='裁剪的最小大小')
    parser.add_argument('--crop_max_size', type=float, default=1.0, help='裁剪的最大大小')
    # 颜色抖动
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT', help='颜色抖动因子（默认：0）')

    '''
    掩蔽策略
    '''
    parser.add_argument('--mask_generator', default='block', type=str, help='块掩蔽或随机掩蔽')
    # 1. 如果使用块掩蔽，设置掩蔽补丁的数量
    parser.add_argument('--num_mask_patches', default=98, type=int, help='需要掩蔽的视觉标记/补丁的数量')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None, help='每个块的最大掩蔽补丁数量')
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16, help='每个块的最小掩蔽补丁数量')
    # 2. 如果使用随机掩蔽，设置掩蔽比例
    parser.add_argument('--ratio_mask_patches', default=None, type=float, help="掩蔽比例")

    '''
    CAE超参数
    '''
    parser.add_argument('--regressor_depth', default=4, type=int, help='回归器的深度')
    parser.add_argument('--decoder_depth', default=4, type=int, help='解码器的深度')
    parser.add_argument('--decoder_embed_dim', default=768, type=int, help='解码器嵌入的维度')
    parser.add_argument('--decoder_num_heads', default=12, type=int, help='解码器的头数')
    parser.add_argument('--decoder_num_classes', default=8192, type=int, help='解码器的类别数，应该与词汇表大小相同')
    parser.add_argument('--decoder_layer_scale_init_value', default=0.1, type=float, help='解码器层级缩放初始值')
    parser.add_argument('--decoder_mlp_ratio', default=4, type=int, help='解码器MLP比例')

    # 解析参数
    args = parser.parse_args()

    return args


def main(args):
    # 设置随机种子
    utils.set_seed(args.seed)

    # 设置 CUDA 的确定性
    cudnn.benchmark = False
    cudnn.deterministic = True

    # 创建模型
    model = create_model(args.model, pretrained=False, num_classes=args.decoder_num_classes)

    # 移动模型到指定设备
    model.to(args.device)

    # 创建优化器
    optimizer = create_optimizer(args.opt, model.parameters(), args)

    # 创建数据集和数据加载器
    dataset_train = build_cae_pretraining_dataset(args)
    # 创建训练数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,  # 数据集，提供训练样本
        batch_size=args.batch_size,  # 批处理大小，从命令行参数中获取
        num_workers=args.num_workers,  # 数据加载时使用的工作线程数
        pin_memory=args.pin_mem,  # 是否固定CPU内存，以便更高效地传输到GPU
        drop_last=True,  # 如果为True，最后一个不完整的批处理将被丢弃
    )

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 进行一次训练
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, epoch, args.device, args
        )

        # 记录训练过程
        if args.output_dir and epoch % args.save_ckpt_freq == 0:
            utils.save_on_master(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))


if __name__ == '__main__':
    args = get_args()
    main(args)
