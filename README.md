# TABS
TABS: Transformer-based Automated Brain Tissue Segmentation

The associated pre-print can be found here: https://arxiv.org/abs/2201.08741

This repository contains the code for TABS, a 3D CNN-Transformer hybrid automated brain tissue segmentation algorithm using T1w structural MRI scans. TABS relies on a Res-Unet backbone, with a Vision Transformer embedded between the encoder and decoder layers.

![fig3_new_small](https://user-images.githubusercontent.com/93843444/150707690-a14744d7-a792-4dc3-b547-a1a63d43c697.jpg)

TABS has demonstrated better performance, generality, and reliability across datasets with different vendors, field strengths, scan parameters, time points, and neuropsychiatric conditions compared to benchmark 3D Unet models. Representative outputs for the three primary datasets tested on are shown below.

![fig4_smlo](https://user-images.githubusercontent.com/93843444/150708119-cee6daf9-88d4-4a6d-8e8a-ea709e2f5c6e.jpg)

## Application Instructions

If you would like to apply TABS to your own dataset, please pre-process the T1w MRI images as follows:

1. Brain Extraction
2. Registration to the isotropic 1 mm MNI152 Space
3. Intensity normalization to a range of -1 to 1
4. Padding/Cropping to 192x192x192

Afterwards, you may use the train and test scripts provided, by changing the arguments and providing your own dataset/dataloader. We have provided pretrained models corresponding to each of the datasets mentioned in the preprint, which can be readily applied.
