import os
import cv2
import torch
import pickle
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
# https://pytorch.apachecn.org/docs/0.3/transforms.html
from sklearn.model_selection import KFold, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, train_test_split
############################################
import albumentations as A
# from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast, \
# https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Resize
# https://github.com/albumentations-team/albumentations_examples
from albumentations.pytorch import ToTensor, ToTensorV2
############################################

def strong_aug2(p=0.5):
    return Compose([
        #RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        ToTensorV2(),
    ], p=p)

def strong_aug(p=0.5):
    return A.Compose([
        A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
        #A.HorizontalFlip(),
        ToTensorV2(),
    ])

def splitFold(x=None, y=None, nfold=5):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=0)
    idx_trva_list = []
    idx_te_list = []
    for idx_tr, idx_te in kf.split(x):
        idx_trva_list.append(idx_tr)
        idx_te_list.append(idx_te)
    idx_list = np.empty([nfold, 3], dtype=object)
    for i in range(nfold):
        idx_list[i][0] = np.setdiff1d(idx_trva_list[i], idx_te_list[(i + 1) % nfold], True)
        idx_list[i][1] = idx_te_list[(i + 1) % nfold]
        idx_list[i][2] = idx_te_list[i]
    return idx_list

def getDataLoader(imageDir, trainBatchSize):
    # TODO: transformer
    trainTransform = strong_aug(0.5)
    validTransform = strong_aug(0.5)
    testTransform = strong_aug(0.5)
    ############################################
    x_dir = os.path.join(imageDir, "train", "x")
    y_dir = os.path.join(imageDir, "train", "y_all")
    imagePaths = np.asarray([os.path.join(x_dir, filename) for filename in os.listdir(x_dir)])
    maskPaths = np.asarray([os.path.join(y_dir, filename[:-4]+".npy") for filename in os.listdir(x_dir)])
    fold_file = "/data1/lzy2020/segpc2021/tmp/fold.pkl"
    if os.path.exists(fold_file):
        with open(fold_file, "rb") as f:
            idx_list = pickle.load(f)
    else:
        idx_list = splitFold(imagePaths, nfold=5)
        with open(fold_file, "wb") as f:
            pickle.dump(idx_list, f)
    fold = 0
    print("using fold #", fold+1)
    idx_tr, idx_va, idx_te = idx_list[0]
    trainImagePaths, trainMaskPaths = imagePaths[idx_tr], maskPaths[idx_tr]
    validImagePaths, validMaskPaths = imagePaths[idx_va], maskPaths[idx_va]
    testImagePaths, testMaskPaths = imagePaths[idx_te], maskPaths[idx_te]
    ############################################
    train_set = segpcDataset(trainImagePaths, trainMaskPaths, trainTransform)
    train_loader = DataLoader(train_set, batch_size=trainBatchSize, num_workers=4, shuffle=True, drop_last=True)
    val_set = segpcDataset(validImagePaths, validMaskPaths, validTransform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=True)
    test_set = segpcDataset(testImagePaths, testMaskPaths, testTransform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=True)
    return train_loader, val_loader, test_loader


class segpcDataset(Dataset):
    def __init__(self, images, masks, transforms):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        image = np.asarray(Image.open(image).convert('RGB'))
        mask = np.load(mask, allow_pickle=True).astype("uint8") / 20
        # (image.shape) (H,W,3)
        # (mask.shape) (H,W)
        """
        image = cv2.imread("image.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        """
        res = {'image': image, 'mask': mask}
        augmented = self.transforms(**res)
        # (image.shape) (3,H,W)
        # (mask.shape) (H,W)
        return augmented
