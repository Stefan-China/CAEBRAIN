import math
import sys
import time
from typing import Iterable

import torch
import torch.nn as nn

import furnace.utils as utils
import torch.nn.functional as F


def train_one_epoch(model: torch.nn.Module, d_vae: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    model.train()  # 设置模型为训练模式
    metric_logger = utils.MetricLogger(delimiter="  ")  # 初始化日志记录器
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录最小学习率
    header = 'Epoch: [{}]'.format(epoch)  # 输出当前的 epoch 信息
    print_freq = 10  # 每 10 步打印一次信息

    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 为每一步分配学习率和权重衰减
        it = start_steps + step  # 全局训练迭代步数
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]  # 更新学习率
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]  # 更新权重衰减

        samples, images, bool_masked_pos = batch  # 从批次中获取样本和图像
        images = images.to(device, non_blocking=True)  # 将图像移动到指定设备
        samples = samples.to(device, non_blocking=True)  # 将样本移动到指定设备
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)  # 将掩码位置移动到指定设备

        with torch.no_grad():
            if args.model_type == 'caev2' and args.discrete_vae_type == 'clip':
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)  # 将掩码位置展平
                input_ids = d_vae.extract_image_features(images)  # 从图像中提取特征
                C = input_ids.shape[-1]  # 获取特征的最后一个维度
                cls_token_labels = input_ids[:, :1, :]  # 获取类标签
                input_ids = input_ids[:, 1:, :]  # 提取输入 ID
                masked_labels = input_ids[bool_masked_pos].reshape(images.shape[0], -1, C)  # 获取被掩盖的标签
                unmasked_labels = input_ids[~bool_masked_pos].reshape(images.shape[0], -1, C)  # 获取未被掩盖的标签
                labels = torch.cat([cls_token_labels, unmasked_labels, masked_labels], dim=1)  # 拼接标签
            else:
                input_ids = d_vae.get_codebook_indices(images).flatten(1)  # 获取代码本索引
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)  # 展平掩码位置
                labels = input_ids[bool_masked_pos]  # 获取被掩盖的标签

        with torch.cuda.amp.autocast():  # 使用混合精度训练
            outputs = model(samples, bool_masked_pos=bool_masked_pos)  # 前向传播
            if len(outputs) == 3:
                outputs, latent_predict, latent_target = outputs  # 获取输出和潜在变量
            else:
                raise NotImplementedError(str(len(outputs)))

            if args.model_type == 'caev2':
                labels = labels / labels.norm(dim=-1, keepdim=True)  # 对标签进行 L2 归一化
                C = outputs.shape[-1]  # 获取输出的最后一个维度
                # 计算可见潜在对齐损失和被掩盖潜在对齐损失
                loss_main = args.latent_alignment_loss_weight * F.cosine_embedding_loss(
                    outputs.reshape(-1, C).float(), labels.reshape(-1, C),
                    torch.ones_like(labels.reshape(-1, C)[:, 0]), reduction="mean"
                )
                loss_align = torch.zeros(1)[0].cuda()  # 初始化对齐损失为零
            else:
                loss_main = nn.CrossEntropyLoss()(input=outputs.float(), target=labels)  # 计算交叉熵损失
                loss_align = args.align_loss_weight * F.mse_loss(
                    latent_predict.float(), latent_target.detach().float(), reduction="mean"
                )  # 计算均方误差损失

            loss = loss_main + loss_align  # 总损失

        loss_value = loss.item()  # 获取损失值
        loss_main_value = loss_main.item()  # 获取主损失值
        loss_align_value = loss_align.item()  # 获取对齐损失值

        if not math.isfinite(loss_value):  # 检查损失值是否为有限值
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)  # 停止训练

        optimizer.zero_grad()  # 清零梯度
        # timm 对某个优化器（adahessian）添加的属性
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)  # 计算梯度范数
        loss_scale_value = loss_scaler.state_dict()["scale"]  # 获取损失缩放值

        torch.cuda.synchronize()  # 同步 CUDA 线程

        if args.model_type != 'caev2':
            mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()  # 计算多标签分类准确率
            metric_logger.update(mlm_acc=mlm_acc)  # 更新日志

            if log_writer is not None:
                log_writer.update(mlm_acc=mlm_acc, head="loss")  # 记录准确率

        metric_logger.update(loss=loss_value)  # 更新损失
        metric_logger.update(loss_main=loss_main_value)  # 更新主损失
        metric_logger.update(loss_align=loss_align_value)  # 更新对齐损失
        metric_logger.update(loss_scale=loss_scale_value)  # 更新损失缩放值

        min_lr = 10.  # 初始化最小学习率
        max_lr = 0.  # 初始化最大学习率
        for group in optimizer.param_groups:  # 遍历优化器的参数组
            min_lr = min(min_lr, group["lr"])  # 更新最小学习率
            max_lr = max(max_lr, group["lr"])  # 更新最大学习率

        metric_logger.update(lr=max_lr)  # 更新最大学习率
        metric_logger.update(min_lr=min_lr)  # 更新最小学习率
        weight_decay_value = None  # 初始化权重衰减值
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]  # 获取权重衰减值
        metric_logger.update(weight_decay=weight_decay_value)  # 更新权重衰减值
        metric_logger.update(grad_norm=grad_norm)  # 更新梯度范数

        if log_writer is not None:  # 如果有日志记录器
            log_writer.update(loss=loss_value, head="loss")  # 记录损失
            log_writer.update(loss=loss_main_value, head="loss_main")  # 记录主损失
            log_writer.update(loss_align_value, head="loss_align")  # 记录对齐损失
            log_writer.update(loss_scale=loss_scale_value, head="opt")  # 记录损失缩放值
            log_writer.update(lr=max_lr, head="opt")  # 记录最大学习率
            log_writer.update(min_lr=min_lr, head="opt")  # 记录最小学习率
            log_writer.update(weight_decay=weight_decay_value, head="opt")  # 记录权重衰减值
            log_writer.update(grad_norm=grad_norm, head="opt")  # 记录梯度范数

            log_writer.set_step()  # 设置当前步数

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)  # 更新学习率调度器

    # 收集所有进程的统计信息
    metric_logger.synchronize_between_processes()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 获取当前时间
    print(now_time, "Averaged stats:", metric_logger)  # 打印平均统计信息
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  # 返回所有指标的平均值
