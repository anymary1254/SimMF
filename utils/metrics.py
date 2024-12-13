import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
    """

    def __init__(self, temperature=0.05, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, zi, zj):
        """
        Calculate NT-Xent loss
        Args:
            zi: first branch's representation
            zj: second branch's representation
        """
        batch_size = zi.shape[0]

        if self.use_cosine_similarity:
            # Flatten the spatial dimensions
            zi_flat = zi.view(zi.shape[0], zi.shape[1], -1)
            zj_flat = zj.view(zj.shape[0], zj.shape[1], -1)

            # Normalize the vectors
            zi_norm = F.normalize(zi_flat, dim=1)
            zj_norm = F.normalize(zj_flat, dim=1)

            # Calculate similarity matrix
            sim_matrix = torch.matmul(zi_norm.transpose(1, 2), zj_norm) / self.temperature

        else:
            zi_flat = zi.view(zi.shape[0], -1)
            zj_flat = zj.view(zj.shape[0], -1)
            sim_matrix = torch.matmul(zi_flat, zj_flat.T) / self.temperature

        # Create labels for positive pairs
        labels = torch.arange(batch_size).to(zi.device)

        # Calculate loss
        loss = self.criterion(sim_matrix, labels)

        return loss
def get_MSE(pred, real):
    return np.mean(np.power(real - pred, 2))

def get_MAE(pred, real):
    return np.mean(np.abs(real - pred))

def get_MAPE(pred, real):
    mapes = []
    for i in range(len(pred)):
        gt_sum = np.sum(np.abs(real[i]))
        er_sum = np.sum(np.abs(real[i] - pred[i]))
        mapes.append(er_sum / gt_sum)
    return np.mean(mapes)

