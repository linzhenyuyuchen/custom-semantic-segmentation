import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
# https://pytorch.apachecn.org/docs/0.3/transforms.html
############################################
# from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast, \
#     Normalize
# from albumentations.pytorch import ToTensor
############################################

def getDataLoader(imageDir, trainBatchSize):
    # TODO: transformer
    trainTransform = transforms.Compose([
        transforms.ToTensor(),
    ])
    validTransform = transforms.Compose([
        transforms.ToTensor(),
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ############################################
    trainDir = os.path.join(imageDir, "C058_train")
    validDir = os.path.join(imageDir, "C058_test")
    trainImagePaths = []
    trainMaskPaths = []
    validImagePaths = []
    validMaskPaths = []
    trainCTDir = os.path.join(trainDir, "CT_origin")
    validCTDir = os.path.join(validDir, "CT_origin")
    trainROIDir = os.path.join(trainDir, "ROI_origin")
    validROIDir = os.path.join(validDir, "ROI_origin")
    for filename in os.listdir(trainCTDir):
        trainImagePaths.append(os.path.join(trainCTDir, filename))
        trainMaskPaths.append(os.path.join(trainROIDir, filename))
    for filename in os.listdir(validCTDir):
        validImagePaths.append(os.path.join(validCTDir, filename))
        validMaskPaths.append(os.path.join(validROIDir, filename))
    ############################################
    train_set = npcTrainDataset(trainImagePaths, trainMaskPaths, trainTransform)
    train_loader = DataLoader(train_set, batch_size=trainBatchSize, num_workers=4, shuffle=True, drop_last=True)

    val_set = npcTestDataset(validImagePaths, validMaskPaths, validTransform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=True)

    test_set = npcTestDataset(validImagePaths, validMaskPaths, testTransform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)

    return train_loader, val_loader, test_loader

def get_array_from_itk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

class npcTrainDataset(Dataset):
    def __init__(self, images, masks, transforms):
        self.depth = 120
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return self.depth*len(self.images)

    def __getitem__(self, index):
        index_ct = index // self.depth
        index_depth = index % self.depth
        img3d = get_array_from_itk(self.images[index_ct])
        mask3d = get_array_from_itk(self.masks[index_ct])
        # (image.shape) (D,H,W) (120, 100, 100)
        # (mask.shape) (D,H,W) (120, 100, 100)
        img2d = img3d[index_depth]
        mask2d = mask3d[index_depth]
        # tmp = np.zeros((2, mask2d.shape[0], mask2d.shape[1]))
        # tmp[0] = 1- mask2d
        # tmp[1] = mask2d

        image_tensor = torch.FloatTensor(img2d).unsequeeze(0)
        mask_tensor = torch.LongTensor(mask2d.astype("uint8"))
        # (image.shape) (1,H,W) (1, 100, 100)
        # (mask.shape) (H,W) (100, 100)
        return {'image': image_tensor, 'mask': mask_tensor }

class npcTestDataset(Dataset):
    def __init__(self, images, masks, transforms):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img3d = get_array_from_itk(self.images[index])
        mask3d = get_array_from_itk(self.masks[index])

        image_tensor = torch.FloatTensor(img3d).permute(0,3,1,2).unsequeeze(1)
        mask_tensor = torch.LongTensor(mask3d.astype("uint8"))
        # (image.shape) (D,1,H,W)
        # (mask.shape) (D,H,W)
        return {'image': image_tensor, 'mask': mask_tensor}
