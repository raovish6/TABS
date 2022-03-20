import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import random
from medpy.metric.binary import hd
import nibabel as nib
import os

from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import jaccard_score

from Models.TABS_Model import TABS

parser = argparse.ArgumentParser()

# Root directory
parser.add_argument('--root', default='', type=str)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--batch_size', default=3, type=int)

parser.add_argument('--test_fold', default=1, type=int)

parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()

def get_loss(model, criterion, mri_images, targets, mode):

    if mode == 'val':
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Gen model outputs
    output = model(mri_images)

    # Construct binary brain map to consider loss only within there
    input_squeezed = torch.squeeze(mri_images,dim=1)
    brain_map = (input_squeezed > -1).float()
    stacked_brain_map = torch.stack([brain_map, brain_map, brain_map], dim=1)

    # Zero out the background of the segmentation output
    isolated_images = torch.mul(stacked_brain_map, output)

    # Calculate loss over just the brain map
    loss = criterion(isolated_images, targets)
    num_brain_voxels = stacked_brain_map.sum()
    loss = loss / num_brain_voxels

    return loss, isolated_images, stacked_brain_map

def tissue_wise_probability_metrics(isolated_image, target, stacked_brain_map):
    criterion = nn.MSELoss()
    criterion = criterion.cuda(args.gpu)

    # metrics dict to store metric for each tissue type
    metrics_list = ['pearson_corr', 'spearman_corr', 'mse']
    metrics = { i : [] for i in metrics_list }

    # list of flattened tensors being collected and their corresponding dict
    necessary_flattened_tensors = ['GT_flattened_0', 'GT_flattened_1', 'GT_flattened_2', 'iso_flattened_0', 'iso_flattened_1', 'iso_flattened_2']
    flattened_tensors = { i : {} for i in necessary_flattened_tensors }

    # flattened single channel brain mask (192x192x192 --> flat)
    mask_flattened = torch.flatten(stacked_brain_map[0])

    # Only save the part of the flattened GT/output that correspond to nonzero values of the brain mask
    for i in range(0,3):
        # flatten gt of channel i (each channel corresponds to a tissue type)
        flattened_tensors['GT_flattened_' + str(i)] = torch.flatten(target[i])
        # choose only the portion of the flattened gt that correspons to the brain
        flattened_tensors['GT_flattened_' + str(i)] = flattened_tensors['GT_flattened_' + str(i)][mask_flattened.nonzero(as_tuple=True)]
        # make this now a numpy array
        flattened_tensors['GT_flattened_' + str(i)] = flattened_tensors['GT_flattened_' + str(i)].cpu().detach().numpy()

        # repeat for the model output image
        flattened_tensors['iso_flattened_' + str(i)] = torch.flatten(isolated_image[i])
        flattened_tensors['iso_flattened_' + str(i)] = flattened_tensors['iso_flattened_' + str(i)][mask_flattened.nonzero(as_tuple=True)]
        flattened_tensors['iso_flattened_' + str(i)] = flattened_tensors['iso_flattened_' + str(i)].cpu().detach().numpy()

    for i in range(0,3):
        # get output and gt from dict i just constructed
        model_output = flattened_tensors['iso_flattened_' + str(i)]
        GT = flattened_tensors['GT_flattened_' + str(i)]

        # get metrics using the numpy arrays of both (cropped to brain)
        cur_pcorr = np.corrcoef(model_output, GT)[0][1]
        cur_scorr = spearmanr(model_output, GT)[0]

        cur_mse = criterion(torch.tensor(model_output).cuda(args.gpu), torch.tensor(GT).cuda(args.gpu))

        metrics['pearson_corr'].append(cur_pcorr)
        metrics['spearman_corr'].append(cur_scorr)
        metrics['mse'].append(cur_mse.item())

    return metrics

def tissue_wise_map_metrics(isolated_image, target, stacked_brain_map):
    # metrics dict to store metric for each tissue type
    metrics_list = ['DICE', 'HD', 'Jaccard']
    metrics = { i : [] for i in metrics_list }

    # list of flattened tensors (segmentation masks) I'm gonna collect and their corresponding dict
    necessary_masks_list = ['GT_0', 'GT_1', 'GT_2', 'iso_0', 'iso_1', 'iso_2']
    necessary_tensors = { i : {} for i in necessary_masks_list }

    # current output and gt is 3x192x192x192. Basically, each voxel of the brain has 3 probabilities assigned to it for each tissue type. Taking the argmax gives us the most likely tissue type of each voxel (now 1x192x192x192)
    full_map_model = torch.argmax(isolated_image,0)
    full_map_GT = torch.argmax(target,0)
    mask = stacked_brain_map[0]
    mask_flattened = torch.flatten(stacked_brain_map[0])

    for i in range(0,3):
        # now that we have the argmax, we can imagine the brain with each voxel having a value of 0,1,2. To get the masks for each tissue type, we save a new tensor corresponding to 1 where the argmax tensor has a value of the given tissue type and 0 otherwise.
        necessary_tensors['GT_' + str(i)] = (full_map_GT==i).float()
        necessary_tensors['iso_' + str(i)] = (full_map_model==i).float()
        if i == 0:
            # make sure background is 0
            necessary_tensors['GT_' + str(i)] = torch.mul(necessary_tensors['GT_' + str(i)], mask)
            necessary_tensors['iso_' + str(i)] = torch.mul(necessary_tensors['iso_' + str(i)], mask)

        # calc HD with the segmentation masks
        h_dist = hd(necessary_tensors['iso_' + str(i)].cpu().detach().numpy(), necessary_tensors['GT_' + str(i)].cpu().detach().numpy())
        metrics['HD'].append(h_dist)

        # now make cropped 1d numpy arrays only containing mask values for within the brain for dice calculation
        necessary_tensors['GT_' + str(i)] = torch.flatten(necessary_tensors['GT_' + str(i)])
        necessary_tensors['GT_' + str(i)] = necessary_tensors['GT_' + str(i)][mask_flattened.nonzero(as_tuple=True)]
        necessary_tensors['GT_' + str(i)] = necessary_tensors['GT_' + str(i)].cpu().detach().numpy()
        necessary_tensors['iso_' + str(i)] = torch.flatten(necessary_tensors['iso_' + str(i)])
        necessary_tensors['iso_' + str(i)] = necessary_tensors['iso_' + str(i)][mask_flattened.nonzero(as_tuple=True)]
        necessary_tensors['iso_' + str(i)] = necessary_tensors['iso_' + str(i)].cpu().detach().numpy()

    for i in range(0,3):
        model_output = necessary_tensors['iso_' + str(i)]
        GT = necessary_tensors['GT_' + str(i)]
        # dice formula
        dice = np.sum(model_output[GT==1])*2.0 / (np.sum(model_output) + np.sum(GT))
        jaccard = jaccard_score(GT, model_output)

        metrics['DICE'].append(dice)
        metrics['Jaccard'].append(jaccard)

    return metrics

if __name__ == '__main__':

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = TABS()

    # load directory
    load_dir = ''

    checkpoint = torch.load(load_dir, map_location=torch.device(args.gpu))
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda(args.gpu)

    # *************************************************************************
    # Place train and validation datasets/dataloaders here
    # *************************************************************************

    criterion = nn.MSELoss(reduction='sum')
    criterion = criterion.cuda(args.gpu)

    probability_metrics_list = ['pearson_corr', 'spearman_corr', 'mse']
    probability_metrics = { i : [] for i in probability_metrics_list }
    map_metrics_list = ['DICE', 'HD', 'Jaccard']
    map_metrics = { i : [] for i in map_metrics_list }

    model.eval()

    with torch.no_grad():
        val_losses = []
        test = []
        val_corr = []

        # Loop through test dataloader here.
        for i  in range(0,1):

            # Sample data for the purpose of demonstration
            mri_images = torch.randn(3,1,192,192,192)
            targets = torch.randn(3,3,192,192,192)

            mri_images = mri_images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            loss, isolated_images, stacked_brain_maps  = get_loss(model, criterion, mri_images, targets, 'val')

            val_losses.append(loss)

            for g in range(0,len(isolated_images)):
                isolated_image = isolated_images[g]
                target = targets[g]
                stacked_brain_map = stacked_brain_maps[g]
                metrics_maps = tissue_wise_map_metrics(isolated_image, target, stacked_brain_map)
                metrics =  tissue_wise_probability_metrics(isolated_image, target, stacked_brain_map)

                for metric in probability_metrics_list:
                    probability_metrics[metric].append(metrics[metric])
                for metric in map_metrics_list:
                    map_metrics[metric].append(metrics_maps[metric])

    val_net_loss = sum(val_losses)/len(val_losses)

    overall_pcorr = probability_metrics['pearson_corr']
    overall_pcorr = np.array(overall_pcorr)
    avg_pcorr = sum(overall_pcorr)/len(overall_pcorr)
    sd_pcorr = np.std(overall_pcorr, axis=0, ddof=1)

    overall_scorr = probability_metrics['spearman_corr']
    overall_scorr = np.array(overall_scorr)
    avg_scorr = sum(overall_scorr)/len(overall_scorr)
    sd_scorr = np.std(overall_scorr, axis=0, ddof=1)

    overall_mse = probability_metrics['mse']
    overall_mse = np.array(overall_mse)
    avg_mse = sum(overall_mse)/len(overall_mse)
    sd_mse = np.std(overall_mse, axis=0, ddof=1)

    overall_DICE = map_metrics['DICE']
    overall_DICE = np.array(overall_DICE)
    avg_DICE = sum(overall_DICE)/len(overall_DICE)
    sd_DICE = np.std(overall_DICE, axis=0, ddof=1)

    overall_HD = map_metrics['HD']
    overall_HD = np.array(overall_HD)
    avg_HD = sum(overall_HD)/len(overall_HD)
    sd_HD = np.std(overall_HD, axis=0, ddof=1)

    overall_jaccard = map_metrics['Jaccard']
    overall_jaccard = np.array(overall_jaccard)
    avg_jaccard = sum(overall_jaccard)/len(overall_jaccard)
    sd_jaccard = np.std(overall_jaccard, axis=0, ddof=1)

    print('Probability-Based Metrics:')
    print('Val Loss: {} | Pearson: {} SD: {} | Spearman: {} SD: {} | MSE: {} SD: {}'.format(val_net_loss, avg_pcorr, sd_pcorr, avg_scorr, sd_scorr, avg_mse, sd_mse))

    print('Map-Based Metrics:')
    print('DICE: {} SD: {} | HD: {} SD: {} | Jaccard: {} SD: {}'.format(avg_DICE, sd_DICE, avg_HD, sd_HD, avg_jaccard, sd_jaccard))

    log_name = os.path.join(args.root, 'test_TABS.txt')
    with open(log_name, "a") as log_file:
        log_file.write('Pearson: {} SD: {} | Spearman: {} SD: {} | MSE: {} SD: {}'.format(avg_pcorr, sd_pcorr, avg_scorr, sd_scorr, avg_mse, sd_mse))
        log_file.write('\n')
        log_file.write('DICE: {} SD: {} | HD: {} SD: {} | Jaccard: {} SD: {}'.format(avg_DICE, sd_DICE, avg_HD, sd_HD, avg_jaccard, sd_jaccard))
        log_file.write('\n')
        log_file.write('pcorr')
        log_file.write('%s\n' % overall_pcorr)
        log_file.write('scorr')
        log_file.write('%s\n' % overall_scorr)
        log_file.write('MSE')
        log_file.write('%s\n' % overall_mse)
        log_file.write('dice')
        log_file.write('%s\n' % overall_DICE)
        log_file.write('jaccard')
        log_file.write('%s\n' % overall_jaccard)
        log_file.write('hd')
        log_file.write('%s\n' % overall_HD)
