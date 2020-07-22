import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin, distance):
        """Triplet loss module
        :param margin: (float) margin parameter for the triplet loss
        :param distance: (function (Tensor, Tensor) -> Tensor) distance function used, e.g. L2 or 1 - cosine"""
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance

    def forward(self, anchor, positive, negative):
        if len(anchor) > 2:
            # Flatten the batch
            # in case the inputs are images (e.g. Atari)
            anchor = anchor.flatten(start_dim=1)
            positive = positive.flatten(start_dim=1)
            negative = negative.flatten(start_dim=1)
        d_AP = self.distance(anchor, positive)
        d_AN = self.distance(anchor, negative)

        loss = torch.clamp(d_AP - d_AN + self.margin, max=0) #torch.max(d_AP - d_AN + self.margin,zero)

        return loss


def l2(a: torch.Tensor, b: torch.Tensor):
    return (a - b).pow(2).sum(1)


def cosine(a: torch.Tensor, b: torch.Tensor):
    return 1 - nn.CosineSimilarity()(a, b)


def make_triplet_loss(type,margin):
    triplet = None
    if type == "cosine":
        triplet = TripletLoss(margin, cosine)
    elif type == "l2":
        triplet = TripletLoss(margin, l2)
    else:
        pass
    return triplet
if __name__ == '__main__':
    a = torch.Tensor([[1,2],[3,4]])
    b = torch.Tensor([[5,6],[7,8]])

    margin = 0.1
    triplet_l2 = TripletLoss(margin, l2)
    triplet_cos = TripletLoss(margin,cosine)


