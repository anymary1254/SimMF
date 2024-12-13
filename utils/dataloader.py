import os
import cv2
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader


class MapData(data.Dataset):
    def __init__(self, datapath, mode='train', road_mode='no', channel=2):
        # 使用固定路径
        root_dir = "/root/SFYinzi/MS-SCL-main"

        # 构建数据路径
        self.datapath = os.path.join(root_dir, datapath, mode)
        self.channel = channel
        self.road_mode = road_mode

        print(f"Loading data from: {self.datapath}")

        if self.channel == 2:
            self.X = np.load(os.path.join(self.datapath, 'X.npy'))
            self.Y = np.load(os.path.join(self.datapath, 'Y.npy'))
        elif self.channel == 1:
            self.X = np.expand_dims(np.load(os.path.join(self.datapath, 'X.npy')), 1)
            self.Y = np.expand_dims(np.load(os.path.join(self.datapath, 'Y.npy')), 1)
        else:
            print('---data channel error!---')

        self.external = np.load(os.path.join(self.datapath, 'ext.npy'))
        if self.channel == 2:
            self.external[:, 0] = self.external[:, 0] * 7
            self.external[:, 1] = self.external[:, 1] * 24
            self.external[:, 4] = self.external[:, 4] * 14

        if self.road_mode == 'xian':  # 添加西安的处理
            print(f'current road map: {self.road_mode}')
            masks_path = os.path.join(root_dir, 'data/masks/xian.npy')
            road_map = np.load(masks_path).astype(np.uint8)
            resized_map = cv2.resize(road_map, (128, 128), interpolation=cv2.INTER_LINEAR)
            self.road_map = np.expand_dims((resized_map > 0.5).astype(np.uint8), axis=0)

        else:
            print('------no road_map INPUT!---------')
            self.road_map = np.zeros((1, 128, 128))

        assert len(self.X) == len(self.Y) and len(self.X) == len(self.external)
        self.len = len(self.X)
        print('# {} samples: {}'.format(mode, len(self.X)))
        cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def __getitem__(self, item):
        x = self.X[item]
        y = self.Y[item]
        ext = self.external[item]

        if self.channel == 2:
            x_in = np.expand_dims(cv2.resize(x[0].copy(), (64, 64)), 0)
            x_out = np.expand_dims(cv2.resize(x[1].copy(), (64, 64)), 0)
            x_new = np.concatenate((x_in, x_out), 0)
        elif self.channel == 1:
            x_new = np.expand_dims(cv2.resize(x[0].copy(), (128, 128)), 0)
        x_new = self.Tensor(x_new)
        y = self.Tensor(y)
        ext = self.Tensor(ext)
        road = self.road_map

        # adaptive weighted road
        mean_A = np.mean(np.abs(x), 0)
        if self.channel == 2:
            mean_A = self.MapCopy(mean_A, 8)  # 128(roadmap)//16(coarse)
        elif self.channel == 1:
            mean_A = self.MapCopy(mean_A, 4)  # 128(roadmap)/32(coarse)
       # mean_A = self.MapCopy(mean_A, 8)
        mean_A = mean_A.reshape((1, mean_A.shape[0], mean_A.shape[1]))
        road = road * mean_A
        road = self.Tensor(road)

        return x_new, ext, y, road

    def __len__(self):
        return self.len

    def MapCopy(self, traffic_map, copy_ratio):
        traffic_up_map = np.zeros((traffic_map.shape[0] * copy_ratio, traffic_map.shape[1] * copy_ratio),
                                  dtype=np.float32)
        for h in range(traffic_map.shape[0]):
            for w in range(traffic_map.shape[1]):
                traffic_up_map[h * copy_ratio: (h + 1) * copy_ratio, w * copy_ratio: (w + 1) * copy_ratio] = \
                    traffic_map[h, w]
        return traffic_up_map


def get_dataloader_sr(data_path, batch_size, mode='train', road='xian', channel=2):
    data = MapData(data_path, mode, road, channel)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=(mode == 'train'))
    return dataloader