import os
import warnings
import numpy as np
import argparse
import math
import warnings
from datetime import datetime
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
from utils.metrics import NTXentLoss


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation)
            for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class MAB(nn.Module):
    def __init__(self, K, d, input_dim, output_dim):
        super(MAB, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=input_dim, units=D, activations=F.relu)
        self.FC_k = FC(input_dims=input_dim, units=D, activations=F.relu)
        self.FC_v = FC(input_dims=input_dim, units=D, activations=F.relu)
        self.FC = FC(input_dims=D, units=output_dim, activations=F.relu)

    def forward(self, Q, K, batch_size):
        query = self.FC_q(Q)
        key = self.FC_k(K)
        value = self.FC_v(K)
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        result = torch.matmul(attention, value)

        result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)
        result = self.FC(result)

        return result


class BottleAttention(nn.Module):
    def __init__(self, K, d, set_dim):
        super(BottleAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1, 1, set_dim, D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(K, d, D, D)
        self.mab1 = MAB(K, d, D, D)

    def forward(self, X):
        batch_size = X.shape[0]
        X = X.flatten(2)
        X = X.unsqueeze(1).permute(0, 1, 3, 2)

        I = self.I.repeat(X.size(0), 1, 1, 1)
        H = self.mab0(I, X, batch_size)
        result = self.mab1(X, H, batch_size)

        result = result.squeeze(1).permute(0, 2, 1).view(batch_size, 128, 32, -1)

        return result


class EnhancedFFTBlock(nn.Module):
    def __init__(self, in_features):
        super(EnhancedFFTBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_features, in_features, 5, 1, 2),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 5, 1, 2),
            nn.BatchNorm2d(in_features)
        )

        self.freq_attention_low = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        self.freq_attention_high = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 8, in_features, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_features * 2, in_features, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True)
        )

    def split_frequency(self, fft_mag, fft_phase, threshold=0.5):
        B, C, H, W = fft_mag.shape
        mask_size = int(min(H, W) * threshold)

        low_mask = torch.ones((B, C, H, W), device=fft_mag.device)
        low_mask[:, :, mask_size:H - mask_size, mask_size:W - mask_size] = 0
        high_mask = 1 - low_mask

        return (fft_mag * low_mask, fft_phase * low_mask), \
            (fft_mag * high_mask, fft_phase * high_mask)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        spatial_out1 = self.conv_block1(x)
        spatial_out2 = self.conv_block2(x)

        fft_feat = torch.fft.rfft2(x, norm='ortho')
        fft_mag = torch.abs(fft_feat)
        fft_phase = torch.angle(fft_feat)

        fft_mag = F.interpolate(fft_mag, size=(H, W), mode='bilinear', align_corners=False)
        fft_phase = F.interpolate(fft_phase, size=(H, W), mode='bilinear', align_corners=False)

        (low_mag, low_phase), (high_mag, high_phase) = self.split_frequency(fft_mag, fft_phase)

        low_freq_info = torch.cat([
            torch.mean(low_mag, dim=1, keepdim=True),
            torch.mean(low_phase, dim=1, keepdim=True)
        ], dim=1)
        low_attention = self.freq_attention_low(low_freq_info)

        high_freq_info = torch.cat([
            torch.mean(high_mag, dim=1, keepdim=True),
            torch.mean(high_phase, dim=1, keepdim=True)
        ], dim=1)
        high_attention = self.freq_attention_high(high_freq_info)

        spatial_out1 = spatial_out1 * low_attention
        spatial_out2 = spatial_out2 * high_attention

        combined = torch.cat([spatial_out1, spatial_out2], dim=1)
        fused = self.fusion(combined)

        channel_weights = self.channel_attention(fused)
        out = fused * channel_weights

        return out + identity
class SimMF(nn.Module):
    def __init__(self, args):
        super(SimMF, self).__init__()
        self.ext_flag = args.ext_flag
        self.map_width = args.map_width
        self.map_height = args.map_height
        self.in_channels = args.channels
        self.out_channels = args.channels

        if self.ext_flag and self.in_channels == 2:  # xian and chengdu
            self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23], ignore 0, thus use 24
            self.embed_weather = nn.Embedding(15, 3)  # 14types: ignore 0, thus use 15

        if self.ext_flag and self.in_channels == 1:  # beijing
            self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23], ignore 0, thus use 24
            self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

        self.ext2lr2 = nn.Sequential(
            nn.Linear(2, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.map_width * self.map_height),
            nn.ReLU(inplace=True)
        )
        self.ext2lr3 = nn.Sequential(
            nn.Linear(3, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.map_width * self.map_height),
            nn.ReLU(inplace=True)
        )
        self.ext2lr4 = nn.Sequential(
            nn.Linear(4, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.map_width * self.map_height),
            nn.ReLU(inplace=True)
        )

        if self.ext_flag:
            conv1_in = self.in_channels + 4
        else:
            conv1_in = self.in_channels
        conv3_in = args.base_channels

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_in, args.base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(args.resnum):
            res_blocks.append(EnhancedFFTBlock(args.base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.down_pool = nn.AvgPool2d(2)

        long_trans = []
        for _ in range(args.attnum):
            long_trans.append(BottleAttention(16, 8, args.point))
        self.long_trans = nn.Sequential(*long_trans)

        # final conv
        self.conv4 = nn.Conv2d(args.base_channels, self.out_channels, 1)

        self.conv_trans = nn.Sequential(
            nn.Conv2d(args.base_channels, args.base_channels, 3, 1, 1),
            nn.BatchNorm2d(args.base_channels),
            nn.ReLU(inplace=True))
        self.loss = NTXentLoss(0.05, True)

    def forward(self, cmap, ext, roadmap):
        # camp:[4, 2, 64, 64]   ext:[4, 5]   roadmap:[4, 1, 128, 128])
        inp = cmap

        # external factor modeling
        if self.ext_flag and self.in_channels == 2:  # XiAn and ChengDu
            ext_out1 = self.embed_day(ext[:, 0].long().view(-1, 1)).view(-1, 2)
            out1 = self.ext2lr2(ext_out1).view(-1, 1, self.map_width, self.map_height)
            ext_out2 = self.embed_hour(ext[:, 1].long().view(-1, 1)).view(-1, 3)
            out2 = self.ext2lr3(ext_out2).view(-1, 1, self.map_width, self.map_height)
            ext_out3 = self.embed_weather(ext[:, 4].long().view(-1, 1)).view(-1, 3)
            out3 = self.ext2lr3(ext_out3).view(-1, 1, self.map_width, self.map_height)
            ext_out4 = ext[:, 2:4]
            out4 = self.ext2lr2(ext_out4).view(-1, 1, self.map_width, self.map_height)
            ext_out = torch.cat([out1, out2, out3, out4], dim=1)  # [4, 4, 64, 64]
            # [4, 2+4, 64, 64]
            inp = torch.cat([cmap, ext_out], dim=1)

        if self.ext_flag and self.in_channels == 1:  # TaxiBJ-P1
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            out1 = self.ext2lr2(ext_out1).view(-1, 1, self.map_width, self.map_height)
            ext_out2 = self.embed_hour(ext[:, 5].long().view(-1, 1)).view(-1, 3)
            out2 = self.ext2lr3(ext_out2).view(-1, 1, self.map_width, self.map_height)
            ext_out3 = self.embed_weather(ext[:, 6].long().view(-1, 1)).view(-1, 3)
            out3 = self.ext2lr3(ext_out3).view(-1, 1, self.map_width, self.map_height)
            ext_out4 = ext[:, :4]
            out4 = self.ext2lr4(ext_out4).view(-1, 1, self.map_width, self.map_height)
            ext_out = torch.cat([out1, out2, out3, out4], dim=1)
            inp = torch.cat([cmap, ext_out], dim=1)

        # input conv
        out1 = self.conv1(inp)  # [4, 2+4, 64, 64] -> [4, 128, 64, 64]

        short_out = self.res_blocks(out1)  # [4, 128, 64, 64]

        out_pool = self.down_pool(short_out)  # avgpooling   [4, 128, 32, 32]
        long_out = self.long_trans(out_pool)  # [4, 128, 32, 32]
        long_out = F.interpolate(long_out, size=list(short_out.shape[-2:]))

        loss = 0

        out = torch.add(long_out, short_out)

        out = self.conv_trans(out)
        out = self.conv4(out)  # [4, 2, 64, 64]

        return out, loss
