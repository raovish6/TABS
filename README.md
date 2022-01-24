# TABS
TABS: Transformer-based Automated Brain Tissue Segmentation

The associated pre-print can be found here: https://arxiv.org/abs/2201.0874

This repository contains the code for TABS, a 3D CNN-Transformer hybrid automated brain tissue segmentation algorithm using T1w structural MRI scans. TABS relies on a Res-Unet backbone, with a Vision Transformer embedded between the encoder and decoder layers.

![fig3_new_small](https://user-images.githubusercontent.com/93843444/150707690-a14744d7-a792-4dc3-b547-a1a63d43c697.jpg)

TABS has demonstrated better performance, generality, and reliability across datasets with different vendors, field strengths, scan parameters, time points, and neuropsychiatric conditions compared to benchmark 3D Unet models. Representative outputs for the three primary datasets tested on are shown below.

![fig4_smlo](https://user-images.githubusercontent.com/93843444/150708119-cee6daf9-88d4-4a6d-8e8a-ea709e2f5c6e.jpg)
