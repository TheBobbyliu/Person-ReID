import torch.nn as nn
import torch
import numpy as np

class CorrLoss(nn.Module):
    def __init__(self):
        super(CorrLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=40)

    def forward(self, feat, targets):
        """
        Args:
            feat: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = feat.size(0)
        feat = feat.view(feat.size(0), -1, 1, 1).squeeze(3).squeeze(2)
        corr = torch.matmul(feat, feat.t())
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        corr_ap, corr_an = [], []
        for i in range(n):
            corr_ap.append(corr[i][mask[i]].min().unsqueeze(0))
            corr_an.append(corr[i][mask[i] == 0].max().unsqueeze(0))
        corr_ap = torch.cat(corr_ap)
        corr_an = torch.cat(corr_an)
        # Compute ranking hinge loss
        y = torch.ones_like(corr_an)
        loss = self.ranking_loss(corr_ap, corr_an, y)
        return loss
