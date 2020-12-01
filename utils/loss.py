import torch
import torch.nn as nn

def dice_score(input,target): # validation
    eps = 0.0001
    input = input.contiguous()
    target = target.contiguous()
    # input = torch.sigmoid(input)
    # input[input >= 0.5] = 1
    # input[input < 0.5] = 0
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps
    union2 = torch.max(input,target).sum() + eps

    iou = inter.float() / union2.float()
    dice = (2 * inter.float()) / union.float()
    return dice.item(), iou.item()

def dice_loss(input,target):
    eps = 0.0001
    input = input.contiguous()
    target = target.contiguous()
    input = torch.sigmoid(input)
    inter = torch.dot(input.view(-1), target.view(-1))

    union = torch.sum(input) + torch.sum(target) + eps

    dice = 1 - (2 * inter.float() + eps) / union.float()

    return dice

class MultiClassDiceLoss(nn.Module):
    """
    input (B, n_classes, H, W)
    target (B, H , W)
    """

    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, inputs, targets, weights=None):
        # targets (B, H, W)
        targets = make_one_hot(targets,n_classes) # (B,n_classes, H, W)
        totalLoss = 0
        for i in range(n_classes):
            diceLoss = dice_loss(inputs[:, i], targets[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
        totalLoss /= n_classes
        return totalLoss

######################################



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

