from collections import defaultdict
import torch.nn as nn
import torch

class EETLoss(nn.Module):
    """
    EET loss.

    Reference:
    WC Furong Xu et al. Metric Learning with Equidistant and Equidistributed Triplet-based Loss for Product Image Search.

    Args:
        margin (float): margin for EET.
    """
    def __init__(self, margin=0.3):
        super(EETLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        same_class_idxes = defaultdict(list)
        for i, pid in enumerate(targets):
            same_class_idxes[pid.unsqueeze(0).cpu().numpy()[0]].append(i)

        # find closest inter-class distance for each class.
        dist_nn = defaultdict(dict)
        for id1, idx1 in same_class_idxes.items():
            for id2, idx2 in same_class_idxes.items():
                if id1 == id2 :
                    continue
                else:
                    dist_nn[id1][id2] = torch.cat([ds[idx2].min().unsqueeze(0) for ds in dist[idx1]]).min().unsqueeze(0)

        loss_e, loss_m = [], []
        for id1, id2_dict in dist_nn.items():
            ids2 = list(id2_dict.keys())
            min_dists2 = torch.cat(list(id2_dict.values()))
            min_idx = torch.argmin(min_dists2)
            min_dist_an = min_dists2[min_idx]
            min_id_an = ids2[min_idx]
            # find minimum of nn'
            min_dist_nn = torch.cat(list(dist_nn[min_id_an].values())).min()
            loss_e.append(torch.abs(min_dist_an - min_dist_nn).unsqueeze(0))
            loss_m.extend([x for x in id2_dict.values()])

        loss_m = torch.cat(loss_m)
        loss_e = torch.cat(loss_e)
        # maximize the minimal distance between classes
        loss_m = torch.exp(-loss_m).mean()
        loss_e = loss_e.mean()
        loss_ec2 = loss_m + loss_e
        
        # loss_mc and loss_ec1
        dist_ap, dist_an, dist_pn = [], [], []
        for i in range(n):
            posidx = torch.argmax(dist[i][mask[i]])
            negidx = torch.argmin(dist[i][mask[i]==0])
            dist_ap.append(dist[i][mask[i]][posidx].unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0][negidx].unsqueeze(0))
            dist_pn.append(dist[mask[i]][posidx][mask[i] == 0][negidx].unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_pn = torch.cat(dist_pn)
        # to stretch the intra-class distance, squeeze the inner-class distance
        y = torch.ones_like(dist_an)
        loss_mc = self.ranking_loss(dist_an, dist_ap, y)
        # to balance the intra-class distances
        loss_ec1 = torch.abs(dist_an - dist_pn).mean()

        loss = loss_ec2 + loss_mc + loss_ec1        
        return loss

