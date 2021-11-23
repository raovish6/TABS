
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from glob import glob
import nibabel as nib
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import random

class MRIDataset():

    def __init__(self, mri_dir, masks_dir, protocol, mode, val_fold, test_fold):
        self.mode = mode
        self.mri_dir = mri_dir
        self.masks_dir = masks_dir
        self.protocol = protocol
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.train_folds = [1,2,3,4,5]

        self.maskFiles = []

        # This dataloader is only for cross validation, but can be adapted for test as well
        assert self.mode in ['train','val', 'test', 'gen_outputs']

        if self.mode == 'train':
            # For this dataset, I did cross validation. If you would like to also have a test fold, uncomment line 35

            self.train_folds.remove(self.test_fold)
            self.train_folds.remove(self.val_fold)

            # Get lists of the image file paths and each code
            self.imageFiles, self.imageCodes = self.getpaths(self.train_folds,self.mri_dir, self.protocol, self.mode)

            # Create dictionary of mask file paths corresponding to each image code
            self.masks_dict = self.getmasks(self.masks_dir, self.protocol, self.imageCodes)

        elif mode == 'val' or mode == 'test' or mode == 'gen_outputs':
            if mode == 'val':
                self.imageFiles, self.imageCodes = self.getpaths([self.val_fold], self.mri_dir, self.protocol, self.mode)

            else:
                self.imageFiles, self.imageCodes = self.getpaths([self.test_fold],self.mri_dir,self.protocol, self.mode)

            self.masks_dict = self.getmasks(self.masks_dir, self.protocol, self.imageCodes)

    def __getitem__(self,idx):
        #this will take in an index that we load the file from
        #let's load the filepath
        image_filepath = self.imageFiles[idx]
        code = self.imageCodes[idx]

        #now we can load the nifti file and process it
        image, stacked_masks = self.loadimage(image_filepath, code, self.protocol)

        sample = {'T1': image,
              'label': stacked_masks,
              }

        if self.mode == 'gen_outputs':
            sample = {'T1': image,
                  'label': stacked_masks,
                  'code': code
                  }

        return sample

    def __len__(self):
        return len(self.imageFiles)

    def getpaths(self, folds, mri_dir, protocol, mode):
        imageFiles = []
        imageCodes = []
        for fold in folds:
            # Add image file paths for every image in a given fold
            imageFiles.extend(sorted(glob(mri_dir + '/Fold_' + str(fold) + '/*.nii.gz')))
        if protocol == 'dlbs':
            for path in imageFiles:
                # The different domains have the codes in slightly different portions of the path
                imageCodes.append(path[-72:-65])
        elif protocol == 'SALD':
            for path in imageFiles:
                imageCodes.append(path[-54:-48])
        elif protocol == 'IXI':
            for path in imageFiles:
                imageCodes.append(path[-59:-56])
        elif protocol == 'total':
            for path in imageFiles:
                if 'dlbs' in path:
                    imageCodes.append(path[-72:-65])
                elif 'SALD' in path:
                    imageCodes.append(path[-54:-48])
                elif 'IXI' in path:
                    imageCodes.append(path[-59:-56])
        # for testing purposes with dlbs
        else:
            for path in imageFiles:
                imageCodes.append(path[-72:-65])
        print("The {} dataset for domain {} now has {} images".format(mode, protocol, len(imageFiles)))
        return imageFiles, imageCodes

    def getmasks(self, masks_dir, protocol, imageCodes):
        intermediate = []
        maskFiles = []
        # Start by making an intermediate list of all the masks paths.
        intermediate.extend(sorted(glob(masks_dir + '/*.nii.gz')))

        for file in intermediate:
            if protocol == 'dlbs':
                cur_code = file[-20:-13]
            if protocol == 'SALD':
                cur_code = file[-19:-13]
            if protocol == 'IXI':
                cur_code = file[-16:-13]
            if protocol == 'total':
                if 'dlbs' in file:
                    cur_code = file[-20:-13]
                if 'SALD' in file:
                    cur_code = file[-19:-13]
                if 'IXI' in file:
                    cur_code = file[-16:-13]

            if cur_code in imageCodes:
                # Only add the mask files that correspond to images we care about
                maskFiles.append(file)

        # Create masks dict
        masks_dict = { i : [] for i in imageCodes }
        for mask in maskFiles:
            for code in imageCodes:
                if code in mask:
                    masks_dict[code].append(mask)
        return masks_dict

    def loadimage(self, image_filepath, code, protocol):
        # Load, splice, and pad image
        image = nib.load(image_filepath)
        image = image.slicer[:,15:207,:]
        image = np.array(image.dataobj)
        image = np.pad(image, [(5, 5), (0, 0), (5,5)], mode='constant')
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        image = image.float()
        if protocol == 'dlbs':
            image = 2*(image / 187) - 1
        if protocol == 'IXI':
            image = 2*(image / 209) - 1
        if protocol == 'SALD' or protocol == 'total':
            image = 2*(image / 250) - 1

        masks = ['mask_1', 'mask_2', 'mask_3']
        mask_tensors = { i : {} for i in masks }

        # Load, slice, and pad each mask. Add them to masks dictionary
        for num in range(1,4):
            mask_tensors['mask_'+ str(num)] = nib.load(self.masks_dict[code][num-1])
            mask_tensors['mask_'+ str(num)] = mask_tensors['mask_'+ str(num)].slicer[:,15:207,:]
            mask_tensors['mask_'+ str(num)] = np.array(mask_tensors['mask_'+ str(num)].dataobj)
            mask_tensors['mask_'+ str(num)] = np.pad(mask_tensors['mask_'+ str(num)], [(5, 5), (0, 0), (5, 5)], mode='constant')
            mask_tensors['mask_'+ str(num)] = torch.from_numpy(mask_tensors['mask_'+ str(num)])

        # Stack all 3 individual brain masks to a single 3 channel GT
        stacked_masks = torch.stack([mask_tensors['mask_1'], mask_tensors['mask_2'], mask_tensors['mask_3']], dim=0)
        stacked_masks = stacked_masks.float()

        return image, stacked_masks
