import random
import numpy as np
import torch
import torch.nn.functional as F

def make_one_hot(gt, num_classes):
    return F.one_hot(gt.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2)


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)