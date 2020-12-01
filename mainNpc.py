import os
import sys
import yaml
import time
import pickle
import random
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
############################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
############################################
import torchvision
import torchvision.utils as vutils
############################################
from torchsummary import summary
############################################
import model as M
import utils.loss as L
from utils.other import *
import dataset.npcDataset as D

def validate(loader, net, criterion, device, validate_args, global_step=0):
    n_classes = validate_args["n_classes"]
    net.eval()
    n_val = len(loader)
    ############################################
    tot_dice = np.zeros(n_classes)
    tot_iou = np.zeros(n_classes)
    count = np.zeros(n_classes)
    total_step = 0
    ############################################
    with tqdm(total=n_val, desc='Validation round', unit='subject', leave=False) as pbar:
        for batch in loader:
            total_step += 1
            imgs = batch["image"]  # (1,m,1,h,w)
            masks_onehot = batch['mask'][0]  # (m,h,w) 
            imgs = imgs.to(device=device, dtype=torch.float32)
            masks_onehot = make_one_hot(masks_onehot, n_classes)  # (m,n_classes,h,w)
            ############################################
            with torch.no_grad():
                masks_pred = np.zeros((imgs.shape[1], 1, imgs.shape[3], imgs.shape[4]))  # (m,1,h,w)
                for i in range(imgs.shape[1]):
                    masks_pred[i] = torch.argmax(net(imgs[:, i]), dim=1).cpu().numpy()  # (1,h,w)
            masks_pred = torch.LongTensor(masks_pred).squeeze(1).cpu()  # (m,h,w)
            masks_pred = make_one_hot(masks_pred, n_classes)  # (m,n_classes,H,W)
            ############################################
            x1 = vutils.make_grid(masks_onehot.float().cpu(), normalize=False, scale_each=True)
            x2 = vutils.make_grid(masks_pred.float().cpu(), normalize=False, scale_each=True)
            writer.add_image(f'mask-{global_step}/gt_mask', x1, global_step=total_step)
            writer.add_image(f'mask-{global_step}/pr_mask', x2, global_step=total_step)
            ############################################
            for c in range(n_classes):
                count[c] += 1
                dice, iou = L.dice_score(masks_pred[:, c], masks_onehot[:, c])
                tot_dice[c] += dice
                tot_iou[c] += iou
            """method 1
            for c in range(n_classes):
                tmpIndex = []
                flag = 0
                for b in range(masks_onehot.shape[0]):
                    if torch.max(masks_onehot[b][c]) > 0:
                        tmpIndex.append(b)
                        flag += 1
                tmpIndex = torch.tensor(tmpIndex)
                if flag > 0:
                    count[c] += 1
                    dice, iou = L.dice_score(masks_pred.index_select(0, tmpIndex)[:, c],
                                           masks_onehot.index_select(0, tmpIndex)[:, c])
                    tot_dice[c] += dice
                    tot_iou[c] += iou
            """
            ############################################
            pbar.update(1)

    ############################################
    for i in range(n_classes):
        tot_dice[i] /= count[i]
        tot_iou[i] /= count[i]

    diceRes, iouRes = np.around(tot_dice, 8), np.around(tot_iou, 8)
    return diceRes, iouRes


def train(train_loader, val_loader, net, criterion, optimizer, device, train_args):
    n_classes, epochs, checkpoint_dir, checkpoint_space = train_args["n_classes"], train_args["epochs"], train_args[
        "checkpoint_dir"], train_args["checkpoint_space"]
    ############################################
    net.train()
    global_step = 1
    ############################################
    with tqdm(total=len(train_loader) * epochs, unit='batch', leave=False) as pbar:
        for epoch in range(epochs):
            pbar.set_description("%d / %d Epochs" % (epoch, epochs))
            epoch_loss = 0
            for batch in train_loader:
                imgs = batch["image"]
                masks = batch['mask']
                # label = batch['label']
                imgs = imgs.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)
                # label = label.to(device=device, dtype=torch.long)
                ############################################
                # masks_pred, label_pred= net(imgs)
                masks_pred = net(imgs)
                loss = criterion(masks_pred, masks.squeeze(1))
                # loss_labels = criterion(label_pred, label.squeeze(1))
                # loss = 0.2*loss_labels + loss_masks
                epoch_loss += loss.item()
                ############################################
                optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                ############################################
                writer.add_scalar('Loss/train_step', loss.item(), global_step)
                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(1)
                ############################################
                global_step += 1
                if global_step % checkpoint_space == 0:
                    net.eval()
                    diceRes, iouRes = validate(val_loader, net, criterion, device, train_args, global_step)
                    net.train()
                    for i in range(n_classes):
                        writer.add_scalar(f'Dice/valid_{i}', diceRes[i], global_step)
                        writer.add_scalar(f'iou/valid_{i}', iouRes[i], global_step)
                    logging.info(f'dice: {diceRes} !')
                    logging.info(f'iou: {iouRes} !')
                    torch.save(net.state_dict(),
                               checkpoint_dir + f'Train_Step_{global_step}.pth')
                    logging.info(f'Checkpoint Step {global_step} saved !')
            writer.add_scalar('Loss/train_epoch', epoch_loss / len(train_loader), epoch)
    ############################################
    writer.close()



def main(args, cfgs):
    in_channels = cfgs["in_channels"]
    n_classes = cfgs["n_classes"]
    multi_gpu = cfgs["gpus"]
    Net = cfgs["net_name"]
    imageDir = cfgs["image_root"]
    epochs = cfgs["epochs"]
    trainBatchSize = cfgs["batch_size_train"]
    checkpoint_dir = cfgs["checkpoint_dir"]
    checkpoint_space = cfgs["checkpoint_space"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ############################################
    validate_flag = args.validate
    model_load_path = args.checkpoint_path
    ############################################
    try:
        if Net == "unet":
            net = M.Unet(in_channels=in_channels, classes=n_classes)
        elif Net == "unetplusplus":
            net = M.UnetPlusPlus(in_channels=in_channels, classes=n_classes)
        elif Net == "linknet":
            net = M.Linknet(in_channels=in_channels, classes=n_classes)
        elif Net == "fpn":
            net = M.FPN(in_channels=in_channels, classes=n_classes)
        elif Net == "pspnet":
            net = M.PSPNet(in_channels=in_channels, classes=n_classes)
        elif Net == "deeplabv3":
            net = M.DeepLabV3(in_channels=in_channels, classes=n_classes)
        elif Net == "deeplabv3plus":
            net = M.DeepLabV3Plus(in_channels=in_channels, classes=n_classes)
        elif Net == "pan":
            net = M.PAN(in_channels=in_channels, classes=n_classes)
        os.environ["CUDA_VISIBLE_DEVICES"] = multi_gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')
        net.to(device=device)
        if len(multi_gpu) > 1:
            net = nn.parallel.DataParallel(net)
        # summary(net, input_size=(3, 512, 512))
        ############################################
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        if n_classes > 1:
            criterion = nn.CrossEntropyLoss()
            # criterion = L.MultiClassDiceLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        ############################################
        train_loader, val_loader, test_loader = D.getDataLoader(imageDir, trainBatchSize)
        logging.info(f"train-set # {len(train_loader)}")
        logging.info(f"valid-set # {len(val_loader)}")
        logging.info(f"test-set # {len(test_loader)}")
        ############################################
        if validate_flag:
            if os.path.exists(model_load_path):
                net.load_state_dict(torch.load(model_load_path, map_location=device))
                logging.info(f'Checkpoint loaded from {model_load_path}')
                validate_args = {"n_classes": n_classes, "checkpoint_dir": checkpoint_dir,
                                 "checkpoint_space": checkpoint_space, }
                diceRes, iouRes = validate(val_loader, net, criterion, device, validate_args)
                logging.info(f'Valid-dataset dice: {diceRes} !')
                logging.info(f'Valid-dataset iou: {iouRes} !')
                diceRes, iouRes = validate(test_loader, net, criterion, device, validate_args)
                logging.info(f'Test-dataset dice: {diceRes} !')
                logging.info(f'Test-dataset iou: {iouRes} !')
            else:
                logging.info(f'No such checkpoint !')
        else:
            # todo lr_scheduler
            #net.load_state_dict(torch.load("/data1/lzy/wsi/checkpoint/deeplab/INTERRUPTED.pth", map_location=device))
            train_args = {"n_classes": n_classes, "epochs": epochs, "checkpoint_dir": checkpoint_dir,
                          "checkpoint_space": checkpoint_space, }
            train(train_loader, val_loader, net, criterion, optimizer, device, train_args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--validate', dest='validate', action="store_true",
                        help='validate or train')
    parser.add_argument('-m', '--checkpoint_path', dest='checkpoint_path', type=str, default="./x.pth",
                        help='validate from this checkpoint')
    return parser.parse_args()

if __name__ == '__main__':
    seed_torch()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    cfgs = yaml.load(open("./config/cfgNpc.yaml", "r"), Loader=yaml.FullLoader)
    args = get_args()

    writer = SummaryWriter(comment=f"--{cfgs['dataset']}-{cfgs['net_name']}-bs{cfgs['batch_size_train']}-n{cfgs['n_classes']}")
    main(args, cfgs)

