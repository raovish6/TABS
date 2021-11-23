# Import libraries
import argparse
import os
import random
import logging
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.TABS.TABS_model import TABS
import torch.distributed as dist
import torch.nn as nn
from MRIDataset import MRIDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--experiment', default='TransBTS', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information

parser.add_argument('--root', default='/media/sail/HDD10T/Vish/Models/segmentation_models', type=str)

parser.add_argument('--MRI_dir_dlbs', default='/media/sail/HDD10T/Vish/Dataset/dlbs', type=str)

parser.add_argument('--GT_dir_dlbs', default='/media/sail/HDD10T/Vish/Dataset/dlbs/Masks_dlbs', type=str)

parser.add_argument('--MRI_dir_SALD', default='/media/sail/HDD10T/Vish/Dataset/SALD', type=str)

parser.add_argument('--GT_dir_SALD', default='/media/sail/HDD10T/Vish/Dataset/SALD/Masks_SALD', type=str)

parser.add_argument('--MRI_dir_IXI', default='/media/sail/HDD10T/Vish/Dataset/IXI_Guys', type=str)

parser.add_argument('--GT_dir_IXI', default='/media/sail/HDD10T/Vish/Dataset/IXI_Guys/Masks_IXI', type=str)

parser.add_argument('--MRI_dir_total', default='/media/sail/HDD10T/Vish/Dataset/total_merged', type=str)

parser.add_argument('--GT_dir_total', default='/media/sail/HDD10T/Vish/Dataset/total_merged/Masks_total', type=str)

# All you need to change. Make sure it matches one of the possible_protocols
parser.add_argument('--protocol', default='total', type=str)

parser.add_argument('--possible_protocols', default=['dlbs', 'SALD', 'IXI', 'total'])

# Choosing MRI sets the number of classes to 3 (3 types of brain tissue), and the input img dim to 192
parser.add_argument('--dataset', default='mri', type=str)

# Data is split into 5 folds. Choose which fold for valiation
parser.add_argument('--val_fold', default=5, type=int)

# I'm doing 5-fold cross validation, but you can easily modify the dataloader to do train/val/test, and add a test dataset/loader into the train code.
parser.add_argument('--test_fold', default=1, type=int)

# Training Information

# I tried 0.0002 and this- this performed a lot better
parser.add_argument('--lr', default=0.00001, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

# Use amsgrad variant of adam optimizer, could play around with this
parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

# List of possible gpus
parser.add_argument('--gpu', default='0,1,2', type=str)

parser.add_argument('--num_workers', default=4, type=int)

# Maximum I found that the gpu could handle
parser.add_argument('--batch_size', default=3, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

# What I found to work best
parser.add_argument('--end_epoch', default=350, type=int)

# Gpu that I am specifically using
parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')

args = parser.parse_args()

def main_worker():

    if args.protocol == 'total':
        args.end_epoch = 200

    assert args.protocol in args.possible_protocols, 'Protocol must be one of 4 possible protocols: dlbs, SALD, IXI, total'

    dirs = { i : [] for i in args.possible_protocols}
    for protocol in args.possible_protocols:
        mri_dir = getattr(args, 'MRI_dir_' + protocol)
        gt_dir = getattr(args, 'GT_dir_' + protocol)
        dirs[protocol].append(mri_dir)
        dirs[protocol].append(gt_dir)

    MRI_dir = dirs[args.protocol][0]
    GT_dir = dirs[args.protocol][1]

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.local_rank)

    # Declare model with conv_patch_representation and learned positional encoding. See TransBTS paper for more info.
    _, model = TransBTS(dataset = args.dataset, _conv_repr=True, _pe_type="learned")

    model.cuda(args.local_rank)

    print(count_parameters(model))

    # Using adam optimizer (amsgrad variant) with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    # MSE loss for this task (regression). Using reduction value of sum because we want to specify the number of voxels to divide by (only in the brain map)
    criterion = nn.MSELoss(reduction = 'sum')
    criterion = criterion.cuda(args.local_rank)

    # Obtain train dataset
    Train_MRIDataset = MRIDataset(MRI_dir, GT_dir, protocol=args.protocol, mode='train', val_fold=args.val_fold, test_fold=args.test_fold)

    # Obtain train dataloader
    Train_dataloader = DataLoader(Train_MRIDataset, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True)

    # Obtain val dataset
    Val_MRIDataset = MRIDataset(MRI_dir, GT_dir, protocol=args.protocol, mode='val', val_fold=args.val_fold, test_fold=args.test_fold)

    # Obtain val_dataloader
    Val_dataloader = DataLoader(Val_MRIDataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True)

    start_time = time.time()

    # Enable gradient calculation for training
    torch.set_grad_enabled(True)

    # Declare lists to keep track of training and val losses over the epochs
    train_global_losses = []
    val_global_losses = []
    best_epoch = 0

    # Main training/validation loop
    for epoch in range(args.start_epoch, args.end_epoch):

        # Declare lists to keep track of losses and metrics within the epoch
        train_epoch_losses = []
        val_epoch_losses = []
        val_epoch_pcorr = []
        val_epoch_psnr = []
        start_epoch = time.time()

        model.train()

        for i, data in enumerate(Train_dataloader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            # Send each image in the batch individually into the model. Save the loss from each one, and update the model at the end of each batch.
            batch_loss = []

            mri_images = data['T1']
            targets = data['label']

            for cur_mri in range(0,len(mri_images)):
                mri_image = mri_images[cur_mri].cuda(args.local_rank, non_blocking=True)
                target = targets[cur_mri].cuda(args.local_rank, non_blocking=True)

                # Get the loss, brain map, and isolated output (background stripped using the brain map). This function fits the model as well.
                loss, isolated_image, stacked_brain_map  = get_loss(model, criterion, mri_image, target, 'train')

                batch_loss.append(loss)
                train_epoch_losses.append(loss)

            # Average the loss over the batch, and update the model accordingly
            loss = sum(batch_loss)/len(batch_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Transition to val mode
        model.eval()

        # Avoid computing gradients during validation to save memory
        with torch.no_grad():

            for i, data in enumerate(Val_dataloader):

                mri_images = data['T1']
                targets = data['label']

                for cur_mri in range(0,len(mri_images)):
                    mri_image = mri_images[cur_mri].cuda(args.local_rank, non_blocking=True)
                    target = targets[cur_mri].cuda(args.local_rank, non_blocking=True)

                    # Get the loss, brain map, and isolated output (background stripped using the brain map). This function fits the model as well.
                    loss, isolated_image, stacked_brain_map = get_loss(model, criterion, mri_image, target, 'val')

                    val_epoch_losses.append(loss)

                    # Calculate the pearson correlation between the output and ground truth only for the voxels of the brain map.
                    cur_pcorr, cur_psnr = overall_metrics(isolated_image, target, stacked_brain_map)
                    val_epoch_pcorr.append(cur_pcorr)
                    val_epoch_psnr.append(cur_psnr)

        end_epoch = time.time()

        # Average train and val loss over every MRI scan in the epoch. Save to global losses which tracks across epochs
        train_net_loss = sum(train_epoch_losses)/len(train_epoch_losses)
        val_net_loss = sum(val_epoch_losses)/len(val_epoch_losses)
        train_global_losses.append(train_net_loss)
        val_global_losses.append(val_net_loss)
        # Average pearson correlation and psnr over the epoch
        psnr = sum(val_epoch_psnr)/len(val_epoch_psnr)
        pcorr = sum(val_epoch_pcorr)/len(val_epoch_pcorr)

        print('Epoch: {} | Train Loss: {} | Val Loss: {} | PSNR: {} | Pearson: {}'.format(epoch,train_net_loss,val_net_loss, psnr, pcorr))

        checkpoint_dir = args.root + '/' + args.protocol
        # Save the model if it reaches a new min validation loss
        if val_global_losses[-1] == min(val_global_losses):
            print('saving model at the end of epoch ' + str(epoch))
            best_epoch = epoch
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}_val_loss_{}.pth'.format(epoch,val_global_losses[-1]))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

    end_time = time.time()
    total_time = (end_time-start_time)/3600
    print('The total training time is {:.2f} hours'.format(total_time))

    print('----------------------------------The training process finished!-----------------------------------')

    train_losses_final = []
    val_losses_final = []
    for i in range(0,len(train_global_losses)):
        train_losses_final.append(train_global_losses[i].item())
        val_losses_final.append(val_global_losses[i].item())

    log_name = os.path.join(args.root, args.protocol, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Loss (%s) ================\n' % now)
        log_file.write('best_epoch: ' + str(best_epoch) + '\n')
        log_file.write('train_losses: ')
        log_file.write('%s\n' % train_losses_final)
        log_file.write('val_losses: ')
        log_file.write('%s\n' % val_losses_final)
        log_file.write('train_time: ' + str(total_time))

    learning_curve(best_epoch, train_global_losses, val_global_losses)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Input the best epoch, list of global (across epoch) train and val loss. Plot learning curve
def learning_curve(best_epoch, train_global_losses, val_global_losses):
    fig, ax1 = plt.subplots(figsize=(12,8))

    ax1.set_xlabel('Epochs')
    ax1.set_xticks(np.arange(0, int(len(train_global_losses)+1), 10))

    ax1.set_ylabel('Loss')
    ax1.plot(train_global_losses, '-r', label = 'training loss', markersize = 3)
    ax1.plot(val_global_losses, '-b', label = 'validation loss', markersize = 3)
    ax1.axvline(best_epoch, color='m', lw=4, alpha=0.5, label='Best epoch')
    ax1.legend(loc = 'upper left')

    plt.savefig(args.root + '/' + args.protocol + '/Learning_Curve_' + args.protocol + '_' + str(args.val_fold) + '.png')

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

# Calculate pearson correlation and psnr only between the voxels of the brain map (do by total brain not tissue type during training)
def overall_metrics(isolated_image, target, stacked_brain_map):
    #  Flatten the GT, isolated output, and brain mask
    GT_flattened = torch.flatten(target)
    iso_flattened = torch.flatten(isolated_image)
    mask_flattened = torch.flatten(stacked_brain_map)

    # Only save the part of the flattened GT/output that correspond to nonzero values of the brain mask
    GT_flattened = GT_flattened[mask_flattened.nonzero(as_tuple=True)]
    iso_flattened = iso_flattened[mask_flattened.nonzero(as_tuple=True)]

    iso_flattened = iso_flattened.cpu().detach().numpy()
    GT_flattened = GT_flattened.cpu().detach().numpy()

    pearson = np.corrcoef(iso_flattened, GT_flattened)[0][1]
    psnr = peak_signal_noise_ratio(iso_flattened, GT_flattened)

    return pearson, psnr

# Given the model, criterion, input, and GT, this function calculates the loss and returns the isolated output (stripped of background) and brain map
def get_loss(model, criterion, mri_image, target, mode):

    if mode == 'val':
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Get brain map from FSL GT by taking all values >0 --> 1
    brain_map = (target>0).float()
    # Take the max projection of this brain map accross all channels to make sure we get all of the voxels with assigned probabilities.
    max_projection, unnecessary = torch.max(brain_map, dim=0, keepdim=True, out=None)
    # Stack this brain map on itself 3 times to make it 3 channels to match output dimensions
    stacked_brain_map = torch.stack([max_projection[0], max_projection[0], max_projection[0]], dim=0)

    # Fit model
    output = model(mri_image)
    isolated_image = output[0][:][:][:][:]

    # Remove background probabilities through element wise multiplication with brain map
    isolated_image = torch.mul(stacked_brain_map, isolated_image)

    # Calc mse sum through the 'sum' reduction and then divide by the number of voxels in the brain map.
    loss = criterion(isolated_image, target)
    num_brain_voxels = stacked_brain_map.sum()
    loss = loss / num_brain_voxels

    return loss, isolated_image, stacked_brain_map

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
