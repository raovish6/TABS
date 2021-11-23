
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

class COBREDataset():

    def __init__(self, mri_dir, masks_dir):
        self.mri_dir = mri_dir
        self.masks_dir = masks_dir

        self.maskFiles = []

        # Get lists of the image file paths and each code
        self.imageFiles, self.imageCodes = self.getpaths(self.mri_dir)

        # Create dictionary of mask file paths corresponding to each image code
        self.masks_dict = self.getmasks(self.masks_dir, self.imageCodes)

    def __getitem__(self,idx):
        #this will take in an index that we load the file from
        #let's load the filepath

        image_filepaths = ['b','b']
        code = self.imageCodes[idx]
        for file in self.imageFiles:
            # print(file)
            run = file.split('-')[-1][1]
            file_code = file.split('-')[1][:-4]
            if code == file_code and run == '1':
                image_filepaths[0] = file
            elif code == file_code and run == '2':
                image_filepaths[1] = file
        #now we can load the nifti file and process it
        loaded_1 = self.loadimage(image_filepaths[0], code, 1)
        loaded_2 = self.loadimage(image_filepaths[1], code, 2)

        sample = {'run_1': loaded_1,
              'run_2': loaded_2,
              }

        return sample

    def __len__(self):
        return len(self.imageCodes)

    def getpaths(self, mri_dir):
        imageFiles = []
        imageCodes = []

        imageFiles.extend(sorted(glob(mri_dir + '/*.nii.gz')))
        for path in imageFiles:
            code = path.split('-')[1][:-4]
            if code not in imageCodes:
                imageCodes.append(code)

        print("The dataset now has {} images".format(len(imageFiles)))
        return imageFiles, imageCodes

    def getmasks(self, masks_dir, imageCodes):
        maskFiles = []
        # Start by making an intermediate list of all the masks paths.
        maskFiles.extend(sorted(glob(masks_dir + '/*.nii.gz')))

        # Create masks dict
        masks_dict = { i : [] for i in imageCodes }
        for code in imageCodes:
            run_1 = []
            run_2 = []
            for file in maskFiles:
                cur_code = file.split('-')[1][:-4]
                cur_run = int(file.split('-')[-1][1])
                if cur_code == code:
                    if cur_run == 1:
                        run_1.append(file)
                    elif cur_run == 2:
                        run_2.append(file)

            masks_dict[code].append([run_1,run_2])
        return masks_dict

    def loadimage(self, image_filepath, code, run):
        # Load, splice, and pad image
        image = nib.load(image_filepath)
        image = image.slicer[1:,15:207,1:]
        image = np.array(image.dataobj)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        image = image.float()
        image = 2*(image / 1004.6309204101562) - 1

        masks = ['mask_1', 'mask_2', 'mask_3']
        mask_tensors = { i : {} for i in masks }

        # Load, slice, and pad each mask. Add them to masks dictionary
        for num in range(1,4):
            mask_tensors['mask_'+ str(num)] = nib.load(self.masks_dict[code][0][run-1][num-1])
            mask_tensors['mask_'+ str(num)] = mask_tensors['mask_'+ str(num)].slicer[1:,15:207,1:]
            mask_tensors['mask_'+ str(num)] = np.array(mask_tensors['mask_'+ str(num)].dataobj)
            mask_tensors['mask_'+ str(num)] = torch.from_numpy(mask_tensors['mask_'+ str(num)])

        # Stack all 3 individual brain masks to a single 3 channel GT
        stacked_masks = torch.stack([mask_tensors['mask_1'], mask_tensors['mask_2'], mask_tensors['mask_3']], dim=0)
        stacked_masks = stacked_masks.float()

        return [image, stacked_masks]
