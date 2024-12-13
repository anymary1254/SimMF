import os
import warnings
import numpy as np
import argparse
import math
import warnings
from datetime import datetime
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils import data
import random
import cv2
import copy
from typing import Optional, List
import torch.nn.functional as F
from torch import nn, Tensor
from PIL import Image
import PIL.ImageOps
from tqdm import tqdm
from model import MSSCL
from utils.dataloader import get_dataloader_sr
from utils.metrics import NTXentLoss, get_MAPE
from utils.util import weights_init_normal, print_model_parm_nums

# 定义缩放实验配置
scaling_experiments = {
    'depth_scaling': [
        {'resnum': 4, 'attnum': 2},  # 0.5x
        {'resnum': 7, 'attnum': 4},  # 1.0x (baseline)
        {'resnum': 14, 'attnum': 8},  # 2.0x
    ],
    'width_scaling': [
        {'base_channels': 64},  # 0.5x
        {'base_channels': 128},  # 1.0x (baseline)
        {'base_channels': 256},  # 2.0x
    ],
    'attention_scaling': [
        {'nheads': 4, 'point': 32},  # 0.5x
        {'nheads': 8, 'point': 64},  # 1.0x (baseline)
        {'nheads': 16, 'point': 128}  # 2.0x
    ]
}

# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=4,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='adam: decay of second order momentum of gradient')
parser.add_argument('--base_channels', type=int, default=128,
                    help='number of feature maps')
parser.add_argument('--seed', type=int, default=2017, help='random seed')
parser.add_argument('--ext_flag', action='store_true',
                    help='whether to use external factors')
parser.add_argument('--map_width', type=int, default=64,
                    help='image width')
parser.add_argument('--map_height', type=int, default=64,
                    help='image height')
parser.add_argument('--channels', type=int, default=2,
                    help='number of flow image channels')
parser.add_argument('--dataset_name', type=str, default='XiAn',
                    help='which dataset to use')
parser.add_argument('--city_road_map', type=str, default='xian',
                    help='which city_road_map to use')
parser.add_argument('--run_num', type=int, default=0,
                    help='save model folder serial number')
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads")
parser.add_argument('--point', default=64, type=int,
                    help="Number of bottleneck point")
parser.add_argument('--resnum', default=7, type=int,
                    help="Number of residual blocks")
parser.add_argument('--attnum', default=4, type=int,
                    help="Number of attention blocks")
parser.add_argument('--scaling_exp', type=str, default='all',
                    help='Scaling experiment to run: depth/width/attention/all')
parser.add_argument('--scaling_factor', type=float, default=1.0,
                    help='Scaling factor relative to baseline')

args = parser.parse_known_args()[0]
args.ext_flag = True


def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    metrics = {
        'mape': 0,
        'inference_time': 0,
        'memory_usage': 0
    }

    total_samples = 0
    start_time = time.time()

    # 记录初始GPU内存
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

    with torch.no_grad():
        for flows_c, ext, flows_f, road in tqdm(dataloader, desc="Evaluating"):
            flows_c = flows_c.to(device)
            ext = ext.to(device)
            flows_f = flows_f.to(device)
            road = road.to(device)

            # 推理
            preds, _ = model(flows_c, ext, road)

            # 计算MAPE
            preds_ = preds.cpu().numpy()
            flows_f_ = flows_f.cpu().numpy()
            metrics['mape'] += get_MAPE(preds_, flows_f_) * len(flows_c)
            total_samples += len(flows_c)

    # 计算平均MAPE
    metrics['mape'] /= total_samples

    # 计算推理时间
    metrics['inference_time'] = (time.time() - start_time) / total_samples

    # 计算GPU内存使用
    if torch.cuda.is_available():
        metrics['memory_usage'] = (torch.cuda.max_memory_allocated() - start_mem) / 1024 ** 2  # MB

    return metrics


def train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, args, device):
    """训练模型的函数"""
    best_mape = float('inf')
    train_metrics = {'mape': [], 'loss': []}

    pbar = tqdm(range(args.n_epochs))
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        epoch_mape = 0

        for i, (real_coarse_A, ext, real_fine_A, road_A) in enumerate(train_dataloader):
            real_coarse_A = real_coarse_A.to(device)
            ext = ext.to(device)
            real_fine_A = real_fine_A.to(device)
            road_A = road_A.to(device)

            optimizer.zero_grad()
            gen_hr, loss1 = model(real_coarse_A, ext, road_A)
            loss = criterion(gen_hr, real_fine_A)
            (loss + loss1).backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mape += get_MAPE(gen_hr.detach().cpu().numpy(),
                                   real_fine_A.cpu().numpy()) * len(real_coarse_A)

        # 计算epoch平均指标
        epoch_loss /= len(train_dataloader)
        epoch_mape /= len(train_dataloader.dataset)

        # 验证
        val_metrics = evaluate_model(model, valid_dataloader, device)

        # 更新进度条
        pbar.set_description(f'Loss: {epoch_loss:.4f}, MAPE: {val_metrics["mape"]:.4f}')

        # 保存最佳模型
        if val_metrics['mape'] < best_mape:
            best_mape = val_metrics['mape']
            torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pt'))

        train_metrics['mape'].append(epoch_mape)
        train_metrics['loss'].append(epoch_loss)

    return train_metrics


def get_experiment_config(exp_name, scaling_factor):
    """根据实验类型和缩放因子返回相应的配置"""
    baseline_configs = {
        'depth_scaling': {'resnum': 7, 'attnum': 4},
        'width_scaling': {'base_channels': 128},
        'attention_scaling': {'nheads': 8, 'point': 64}
    }

    config = baseline_configs[exp_name].copy()
    for key in config:
        config[key] = int(config[key] * scaling_factor)
    return config


def save_config_results(save_path, scaling_factor, config, train_metrics, eval_metrics):
    """保存单个配置的实验结果"""
    with open(save_path, 'w') as f:
        f.write(f"Scaling Factor: {scaling_factor}\n")
        f.write(f"Configuration: {config}\n\n")
        f.write("Training Metrics:\n")
        f.write(f"Final MAPE: {train_metrics['mape'][-1]:.4f}\n")
        f.write(f"Final Loss: {train_metrics['loss'][-1]:.4f}\n\n")
        f.write("Evaluation Metrics:\n")
        for metric, value in eval_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")


def save_analysis_results(save_path, results):
    """保存总体分析结果"""
    with open(save_path, 'w') as f:
        for exp_name, exp_results in results.items():
            f.write(f"\n{'-' * 50}\n")
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"{'-' * 50}\n\n")

            for result in exp_results:
                f.write(f"Scaling Factor: {result['scaling_factor']}\n")
                f.write(f"Configuration: {result['config']}\n")
                f.write("Evaluation Metrics:\n")
                for metric, value in result['evaluation_metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")


def plot_scaling_analysis(results, save_path):
    """绘制缩放分析图表"""
    for exp_name, exp_results in results.items():
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 提取数据
        x = [r['scaling_factor'] for r in exp_results]
        y_mape = [r['evaluation_metrics']['mape'] for r in exp_results]
        y_time = [r['evaluation_metrics']['inference_time'] for r in exp_results]
        y_memory = [r['evaluation_metrics']['memory_usage'] for r in exp_results]

        # 性能图
        ax1.plot(x, y_mape, 'o-', label='MAPE')
        ax1.set_title(f'Performance vs {exp_name} Scaling')
        ax1.set_xlabel('Scaling Factor')
        ax1.set_ylabel('MAPE')
        ax1.grid(True)

        # 资源使用图
        ax2.plot(x, y_time, 's-', label='Inference Time (s)', color='orange')
        ax2.set_xlabel('Scaling Factor')
        ax2.set_ylabel('Inference Time (s)', color='orange')

        # 添加内存使用曲线
        ax3 = ax2.twinx()
        ax3.plot(x, y_memory, '^-', label='Memory Usage (MB)', color='green')
        ax3.set_ylabel('Memory Usage (MB)', color='green')

        # 设置图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{exp_name}_analysis.png'))
        plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建结果保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = os.path.join('model', f'scaling_analysis_{timestamp}')
    os.makedirs(base_save_path, exist_ok=True)

    # 准备数据加载器
    source_datapath = os.path.join('data', args.dataset_name)
    train_dataloader = get_dataloader_sr(source_datapath, args.batch_size, 'train',
                                         args.city_road_map, args.channels)
    valid_dataloader = get_dataloader_sr(source_datapath, args.batch_size, 'valid',
                                         args.city_road_map, args.channels)

    # 确定要运行的实验
    if args.scaling_exp == 'all':
        experiments_to_run = scaling_experiments
    else:
        experiments_to_run = {args.scaling_exp: scaling_experiments[args.scaling_exp]}

    # 存储实验结果
    results = {}

    # 运行实验
    for exp_name, configs in experiments_to_run.items():
        results[exp_name] = []

        for i, config in enumerate(configs):
            # 计算缩放因子
            scaling_factor = 0.5 if i == 0 else (2.0 if i == 2 else 1.0)

            # 创建实验特定的保存路径
            exp_save_path = os.path.join(base_save_path, exp_name, f'factor_{scaling_factor}')
            os.makedirs(exp_save_path, exist_ok=True)

            # 更新参数
            exp_args = copy.deepcopy(args)
            exp_args.save_path = exp_save_path
            for k, v in config.items():
                setattr(exp_args, k, v)

            # 创建模型
            model = MSSCL(exp_args).to(device)
            model.apply(weights_init_normal)

            # 设置优化器和损失函数
            optimizer = torch.optim.Adam(model.parameters(), lr=exp_args.lr,
                                         betas=(exp_args.b1, exp_args.b2))
            criterion = nn.MSELoss()

            print(f"\nRunning {exp_name} experiment with scaling factor {scaling_factor}")
            train_metrics = train_model(model, train_dataloader, valid_dataloader,
                                        optimizer, criterion, exp_args, device)

            # 评估模型
            eval_metrics = evaluate_model(model, valid_dataloader, device)

            # 记录结果
            results[exp_name].append({
                'scaling_factor': scaling_factor,
                'config': config,
                'train_metrics': train_metrics,
                'evaluation_metrics': eval_metrics
            })

            # 保存当前配置的结果
            config_results_path = os.path.join(exp_save_path, 'results.txt')
            save_config_results(config_results_path, scaling_factor, config,
                                train_metrics, eval_metrics)

    # 保存总体分析结果
    analysis_save_path = os.path.join(base_save_path, 'scaling_analysis_results.txt')
    save_analysis_results(analysis_save_path, results)

    # 生成可视化图表
    plot_scaling_analysis(results, base_save_path)

    print(f"\nExperiment results saved in: {base_save_path}")


def main():  # 函数定义
    # 设置设备     # 这里及以下都需要4个空格缩进
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建结果保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_save_path = os.path.join('model', f'scaling_analysis_{timestamp}')
    os.makedirs(base_save_path, exist_ok=True)
if __name__ == "__main__":
    main()