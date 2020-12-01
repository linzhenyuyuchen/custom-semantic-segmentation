import os
import torch
import pickle
import numpy as np
from glob import glob
from PIL import Image
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
    with open("/data1/lzy2020/checkpoint/fold1.pkl", "rb") as f:
        data = pickle.load(f)
    train_dirs, valid_dirs, test_dirs = data["train_dirs"], data["valid_dirs"], data["test_dirs"]
    trainImagePaths = []
    validDirs = []
    testDirs = []
    for subjectDir in train_dirs:
        for fileName in os.listdir(subjectDir):
            filePath = os.path.join(subjectDir, fileName)
            trainImagePaths.append(filePath)
    for subjectDir in valid_dirs:
        validDirs.append(subjectDir)
    for subjectDir in test_dirs:
        testDirs.append(subjectDir)
    """
    n_folds = 5
    imageDir = os.path.join(imageDir, "digestpath_img_patch")
    # maskDir = os.path.join(imageDir, "digestpath_mask_patch")
    # typeNames = ["normal", "low level", "high level"]
    typeNames = ["low level", "high level"]
    trainImagePaths = []
    validDirs = []
    testDirs = []
    validImagePaths = []
    testImagePaths = []
    for i in range(len(typeNames)):
        print(typeNames[i])
        subDir = os.path.join(imageDir, typeNames[i])
        subjectIds = os.listdir(subDir)
        tmpIndex1 = len(subjectIds) // n_folds
        tmpIndex2 = len(subjectIds) // n_folds * 2

        for subjectId in subjectIds[tmpIndex2:]:
            subjectDir = os.path.join(subDir, subjectId)
            for fileName in os.listdir(subjectDir):
                filePath = os.path.join(subjectDir, fileName)
                trainImagePaths.append(filePath)
        for subjectId in subjectIds[:tmpIndex1]:
            subjectDir = os.path.join(subDir, subjectId)
            validDirs.append(subjectDir)
            for fileName in os.listdir(subjectDir):
                filePath = os.path.join(subjectDir, fileName)
                validImagePaths.append(filePath)
        for subjectId in subjectIds[tmpIndex1:tmpIndex2]:
            subjectDir = os.path.join(subDir, subjectId)
            testDirs.append(subjectDir)
            for fileName in os.listdir(subjectDir):
                filePath = os.path.join(subjectDir, fileName)
                testImagePaths.append(filePath)
    print("normal")
    subDir = os.path.join(imageDir, "normal")
    subjectIds = os.listdir(subDir)
    tmpIndex1 = len(subjectIds) // 2
    for subjectId in subjectIds[:tmpIndex1]:
        subjectDir = os.path.join(subDir, subjectId)
        validDirs.append(subjectDir)
        for fileName in os.listdir(subjectDir):
            filePath = os.path.join(subjectDir, fileName)
            validImagePaths.append(filePath)
    for subjectId in subjectIds[tmpIndex1:]:
        subjectDir = os.path.join(subDir, subjectId)
        testDirs.append(subjectDir)
        for fileName in os.listdir(subjectDir):
            filePath = os.path.join(subjectDir, fileName)
            testImagePaths.append(filePath)
    """
    ############################################
    trainMaskPaths = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch_npy_01/")[:-4] + ".npy" for p in
                      trainImagePaths]
    # validMaskPaths = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch_npy/")[:-4]+".npy" for p in validImagePaths]
    # testMaskPaths = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch_npy/")[:-4]+".npy" for p in testImagePaths]
    validMaskDirs = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch_npy_01/") for p in validDirs]
    testMaskDirs = [p.replace("/digestpath_img_patch/", "/digestpath_mask_patch_npy_01/") for p in testDirs]
    ############################################
    train_set = wsiTrainDataset(trainImagePaths, trainMaskPaths, trainTransform)
    train_loader = DataLoader(train_set, batch_size=trainBatchSize, num_workers=4, shuffle=True, drop_last=True)
    # val_set = wsiDataset2(validImagePaths, validMaskPaths, transform_val)
    val_set = wsiTestDataset(validDirs, validMaskDirs, validTransform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=True)
    # test_set = wsiDataset2(testImagePaths, testMaskPaths, transform_val)
    test_set = wsiTestDataset(testDirs, testMaskDirs, testTransform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)
    return train_loader, val_loader, test_loader


class wsiTrainDataset(Dataset):
    def __init__(self, images, masks, transforms):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        image = Image.open(image).convert('RGB')
        image_tensor = self.transforms(image)
        mask = np.load(mask, allow_pickle=True).astype("uint8")
        mask_tensor = torch.LongTensor(mask)
        # (image.shape) (3,H,W)
        # (mask.shape) (H,W)
        return {'image': image_tensor, 'mask': mask_tensor }

class wsiTestDataset(Dataset):
    def __init__(self, imageDirs, maskDirs, transforms):
        self.H = 512
        self.W = 512
        self.C = 3
        self.imageDirs = imageDirs
        self.maskDirs = maskDirs
        self.transforms = transforms
        self.images = []
        self.masks = []
        for sub, sub_mask in zip(self.imageDirs, self.maskDirs):
            tmp1 = []
            tmp2 = []
            for fileName in os.listdir(sub):
                imagePath = os.path.join(sub, fileName)
                maskPath = os.path.join(sub_mask, fileName[:-4]+".npy")
                tmp1.append(imagePath)
                tmp2.append(maskPath)
            self.images.append(tmp1)
            self.masks.append(tmp2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        images, masks = self.images[index], self.masks[index]

        image_tensor = np.zeros((len(images), self.H, self.W, self.C))
        for i in range(len(images)):
            image_tensor[i] = np.asarray(Image.open(images[i]).convert('RGB'))

        mask_tensor = np.zeros((len(masks), self.H, self.W))
        label_tensor = np.zeros(len(masks))
        for i in range(len(masks)):
            mask = np.load(masks[i], allow_pickle=True).astype("uint8")
            mask_tensor[i] = mask
            label_tensor[i] = np.max(mask)

        image_tensor = torch.FloatTensor(image_tensor).permute(0,3,1,2)
        mask_tensor = torch.LongTensor(mask_tensor.astype("uint8"))
        label_tensor = torch.LongTensor(label_tensor.astype("uint8"))
        # (image.shape) (M,C,H,W)
        # (mask.shape) (M,H,W)
        # (label.shape) (M)
        return {'image': image_tensor, 'mask': mask_tensor, "label": label_tensor}
