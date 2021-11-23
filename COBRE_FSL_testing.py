import argparse
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from COBRE_dataloader import COBREDataset
import random
from medpy.metric.binary import hd
import nibabel as nib
import os

from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import jaccard_score
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()

# DataSet Information

parser.add_argument('--MRI_dir', default="/media/sail/HDD10T/Vish/data_organized/COBRE/dataset/Repeated_scans", type=str)

parser.add_argument('--root', default="/media/sail/HDD10T/Vish/data_organized/COBRE/dataset/", type=str)

parser.add_argument('--masks_dir', default="/media/sail/HDD10T/Vish/data_organized/COBRE/dataset/FAST_pve", type=str)

# only thing to change
parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--num_workers', default=0, type=int)

parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--test_fold', default=1, type=int)

parser.add_argument('--gpu', default=2, type=int)

args = parser.parse_args()

def get_brain_map(run_1, run_2):

    # to get reproducible resutls
    # Get brain map from FSL GT by taking all values 0 --> 1
    brain_map_1 = (run_1>0).float()
    brain_map_2 = (run_2>0).float()
    # Take the max projection of this brain map accross all channels to make sure we get all of the voxels with assigned probabilities (3x192x192x192 --> 1x192x192x192).
    max_projection_1, unnecessary = torch.max(brain_map_1, dim=0, keepdim=True, out=None)
    max_projection_2, unnecessary = torch.max(brain_map_2, dim=0, keepdim=True, out=None)

    max_projection = max_projection_1.logical_or(max_projection_2)

    # Stack this brain map on itself 3 times to make it 3 channels to match output dimensions (after stacking, 3x192x192x192)
    stacked_brain_map = torch.stack([max_projection[0], max_projection[0], max_projection[0]], dim=0)

    return stacked_brain_map

def tissue_wise_probability_metrics(isolated_image, target, stacked_brain_map):
    criterion = nn.MSELoss()
    criterion = criterion.cuda(args.gpu)

    # metrics dict to store metric for each tissue type
    metrics_list = ['pearson_corr', 'spearman_corr', 'psnr', 'mse']
    metrics = { i : [] for i in metrics_list }

    # list of flattened tensors I'm gonna collect and their corresponding dict
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
        cur_psnr = peak_signal_noise_ratio(model_output, GT)

        cur_mse = criterion(torch.tensor(model_output).cuda(args.gpu), torch.tensor(GT).cuda(args.gpu))

        metrics['pearson_corr'].append(cur_pcorr)
        metrics['spearman_corr'].append(cur_scorr)
        metrics['psnr'].append(cur_psnr)
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

    Test_MRIDataset = COBREDataset(args.MRI_dir, args.masks_dir)

    Test_dataloader = DataLoader(Test_MRIDataset, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True)

    probability_metrics_list = ['pearson_corr', 'spearman_corr', 'psnr', 'mse']
    probability_metrics = { i : [] for i in probability_metrics_list }
    map_metrics_list = ['DICE', 'HD', 'Jaccard']
    map_metrics = { i : [] for i in map_metrics_list }

    with torch.no_grad():

        count = 0
        for i, data in enumerate(Test_dataloader):

            run_1 = data['run_1']
            run_2 = data['run_2']

            fsl_1 = run_1[1][0]
            fsl_2 = run_2[1][0]

            stacked_brain_map = get_brain_map(fsl_1, fsl_2)

            metrics_maps = tissue_wise_map_metrics(fsl_1, fsl_2, stacked_brain_map)
            metrics =  tissue_wise_probability_metrics(fsl_1, fsl_2, stacked_brain_map)

            for metric in probability_metrics_list:
                probability_metrics[metric].append(metrics[metric])
            for metric in map_metrics_list:
                map_metrics[metric].append(metrics_maps[metric])

    overall_pcorr = probability_metrics['pearson_corr']
    overall_pcorr = np.array(overall_pcorr)
    avg_pcorr = sum(overall_pcorr)/len(overall_pcorr)

    overall_scorr = probability_metrics['spearman_corr']
    overall_scorr = np.array(overall_scorr)
    avg_scorr = sum(overall_scorr)/len(overall_scorr)

    overall_psnr = probability_metrics['psnr']
    overall_psnr = np.array(overall_psnr)
    avg_psnr = sum(overall_psnr)/len(overall_psnr)

    overall_mse = probability_metrics['mse']
    overall_mse = np.array(overall_mse)
    avg_mse = sum(overall_mse)/len(overall_mse)

    overall_DICE = map_metrics['DICE']
    overall_DICE = np.array(overall_DICE)
    avg_DICE = sum(overall_DICE)/len(overall_DICE)

    overall_HD = map_metrics['HD']
    overall_HD = np.array(overall_HD)
    avg_HD = sum(overall_HD)/len(overall_HD)

    overall_jaccard = map_metrics['Jaccard']
    overall_jaccard = np.array(overall_jaccard)
    avg_jaccard = sum(overall_jaccard)/len(overall_jaccard)

    print('Probability-Based Metrics:')
    print('Pearson: {} | Spearman: {} | psnr: {} | MSE: {} |'.format(avg_pcorr, avg_scorr, avg_psnr, avg_mse))

    print('Map-Based Metrics:')
    print('DICE: {} | HD: {} | Jaccard: {}'.format(avg_DICE, avg_HD, avg_jaccard))

    log_name = os.path.join(args.root, 'COBRE_FSL.txt')
    with open(log_name, "a") as log_file:
        log_file.write('Pearson: {} | Spearman: {} | psnr: {} | MSE: {} |'.format(avg_pcorr, avg_scorr, avg_psnr, avg_mse))
        log_file.write('\n')
        log_file.write('DICE: {} | HD: {} | Jaccard: {}'.format(avg_DICE, avg_HD, avg_jaccard))
