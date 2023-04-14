# TABS
TABS: Transformer-based Automated Brain Tissue Segmentation

The associated pre-print can be found here: https://arxiv.org/abs/2201.08741

This repository contains the code for TABS, a 3D CNN-Transformer hybrid automated brain tissue segmentation algorithm using T1w structural MRI scans. TABS relies on a Res-Unet backbone, with a Vision Transformer embedded between the encoder and decoder layers.

![fig3_new_small](https://user-images.githubusercontent.com/93843444/150707690-a14744d7-a792-4dc3-b547-a1a63d43c697.jpg)

TABS has demonstrated better performance, generality, and reliability across datasets with different vendors, field strengths, scan parameters, time points, and neuropsychiatric conditions compared to benchmark 3D Unet models. Representative outputs for the three primary datasets tested on are shown below.

![fig4_smlo](https://user-images.githubusercontent.com/93843444/150708119-cee6daf9-88d4-4a6d-8e8a-ea709e2f5c6e.jpg)

## Data Preparation

If you would like to apply TABS for tissue segmentation to your own dataset, please pre-process the T1w MRI images as follows:

1. Brain Extraction
2. Registration to the isotropic 1 mm MNI152 Space
3. Intensity normalization to a range of -1 to 1
4. Padding/Cropping to 192x192x192

Afterwards, you may use the train and test scripts provided, by changing the arguments and providing your own dataset/dataloader. When being sent to the model, each image should be in the shape: (batch_size, channel, dim, dim, dim).

When finetuning TABS for other tasks, it is compatible with other image dimensions divisible by 16.

## Quick Start

Clone repository from github
```
git init
git clone https://github.com/raovish6/TABS
```

Download pretrained weights
```
pip install gdown
gdown https://drive.google.com/uc?id=1Du6N8hr4lcRCjwSYuwLsepzWVXPmdjEr
```

Import within python
```python
from Models.TABS_Model import TABS
import torch

model = TABS()
checkpoint = torch.load('./best_model_TABS.pth', map_location=torch.device(0))
model.load_state_dict(checkpoint['state_dict'])

example = torch.rand(1,1,192,192,192)
with torch.no_grad():
  output = model(example)
print(output.shape)
```

## Model Parameters

If fine tuning the model, alter these model parameters when calling the model.
```
TABS()
positional arguments:
  img_dim (default = 192)                Input image dimension
  patch_dim (default = 8)                Patch dimension (for transformer)
  img_ch (default = 1)                   Input image channels
  output_ch (default = 3)                Input output channels
  embedding_dim (default = 512)          Input embedding dimension (for transformer)
  num_heads (default = 8)                Number of transformer heads
  num_layers (default = 4)               Number of transformer layers
  dropout_rate (default = 0.1)           Dropout rate
  attn_dropout_rate (default = 0.1)      Dropout rate for attention
```

## Citation

Please cite this paper if you use our code or model in your work or find it useful:

```
AUTHOR=Rao Vishwanatha M., Wan Zihan, Arabshahi Soroush, Ma David J., Lee Pin-Yu, Tian Ye, Zhang Xuzhe, Laine Andrew F., Guo Jia
TITLE=Improving across-dataset brain tissue segmentation for MRI imaging using transformer  	
JOURNAL=Frontiers in Neuroimaging     
VOLUME=1      	
YEAR=2022   
URL=https://www.frontiersin.org/articles/10.3389/fnimg.2022.1023481       
DOI=10.3389/fnimg.2022.1023481    
ISSN=2813-1193  
```
