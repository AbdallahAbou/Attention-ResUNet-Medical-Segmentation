# Attention Residual UNet for Liver and Hepatic Vessel Segmentation

This repository contains the code used for liver and hepatic vessel segmentation from CT images, developed as part of a research paper on segmentation models using Attention Residual UNet. The project leverages the Medical Decathlon dataset and focuses on accurately segmenting the liver and hepatic vessels from CT scans.

## Project Structure

```
abouabdallah_forschungsphase1/
├── CT_Viewer_Trame/
│   ├── models/
│   │   ├── liver_model_classification.pth
│   │   └── vessel_model_classification_16.pth
│   ├── modules/
│   │   ├── __pycache__/
│   │   ├── AttentionResUnet.py
│   │   ├── func.py
│   │   ├── model_loader.py
│   │   └── app.py
│   ├── docs/
│   ├── download/
│   ├── job/
│   ├── models/
│   ├── notebooks/
│   ├── output/
│   │   ├── liver_model_classification.pth
│   │   └── vessel_model_classification_16.pth
│   └── src/
│       ├── data/
│       ├── models/
│       │   ├── __init__.py
│       │   ├── .gitkeep
│       │   ├── attention_res_unet.py
│       │   ├── pred_model.py
│       │   ├── test_model.py
│       │   └── train_model.py
│       └── visualisation/
│           ├── .gitkeep
│           └── visualise.py
├── image-1.png
├── image-2.png
├── image-3.png
├── image.png
├── LICENSE
├── main.py
├── notebook.ipynb
├── README.md
├── setup.sh
```

## CT Viewer Trame App

The CT Viewer Trame App is an interactive visualization tool that allows users to view and analyze CT scan images with automated segmentation capabilities. The application uses deep learning models to segment the liver, hepatic vessels, and tumors in CT scans, helping radiologists and clinicians in diagnosis and treatment planning.

### Features

- Interactive CT scan visualization with multi-planar reconstruction (axial, sagittal, and coronal views)
- Automated segmentation of liver, vessels, and tumors using pre-trained Attention Residual UNet models
- Ability to toggle visibility of different segmentation overlays
- Synchronized slice navigation across all views
- DICOM/NIFTI file support for medical imaging data

### Usage

1. Upload a CT scan image in NIFTI format (.nii.gz)
2. Run segmentation with a single click
3. Toggle visibility of different segmentation elements (liver, vessels, tumor)
4. Navigate through slices using the position slider

![CT Viewer Upload Interface](image-2.png)
*Figure 1: CT Viewer interface showing the file upload and segmentation controls*

![CT Viewer Segmentation View](image-3.png)
*Figure 2: Multi-planar reconstruction view with segmentation overlays and navigation controls*

### Implementation

The application is built using:
- Trame framework for web-based visualization
- PyTorch for deep learning model inference
- NiBabel for medical image I/O
- Attention Residual UNet architecture for segmentation

The models are pre-trained on the Medical Segmentation Decathlon dataset and can achieve high accuracy in identifying liver tissue, vascular structures, and potential tumors.

### Folder structure

```bash
.                                                                                       
├── data                      <- Data dirs should NOT be checked into git
│   └── raw                   <- The final dataset for model        
├── docs                      <- Documentation for project 
├── .gitignore                <- Ignore files that should not be commited 
├── main.py                   <- This file will be called when running a slurm job.
├── models                    <- Trained models, model predictions or model summaries.
├── notebooks                 <- Jupyter notebooks. 
├── README.md                 <- README file for this git repo.
├── requirements.txt          <- Pip modules to be installed. See "pip install -r".
├── RUN_FIRST.sh              <- RUN THIS FIRST! uncaches data dirs.
├── setup.sh                  <- This file will be called before *main.py* when running a slurm job.
└── src                       <- Source code for this project.
    ├── __init__.py           <- Makes src a python module.
    ├── data                  <- Scripts to download and process data.
    ├── models                <- Scripts to train models and make predictions.
    └── visualisation         <- scripts to create visualisations
```

### RUN_FIRST.sh

Please run the following command before making any changes to the directory:

```bash
bash RUN_FIRST.sh
```

This script will untrack the data and output directories, so they don't get checked into the remote repository, and then delete itself.

### Data storage

As git should not be used to store large amounts of unchanging data, please do not check image files into the repository. Once you have run **RUN_FIRST.sh** You may use the *data* directory to locally store images you have downloaded from elsewhere.

## Installation


### Data

The project utilizes the Medical Segmentation Decathlon dataset, specifically:

- **Liver Segmentation**: Located at `/download/dataset/raw/Task03_Liver`
- **Hepatic Vessel Segmentation**: Located at `/download/dataset/raw/Task08_HepaticVessel`

The data is downloaded automatically. Source: http://medicaldecathlon.com/ . 

### Requirements

Make sure to install the required dependencies.

```bash
pip install -r requirements.txt

```

## Usage


### Training and Evaluation:

To train and evaluate the model, run the main script:

```bash
python main.py --ScratchDir ./job --DownloadDir ./download --OutputDir ./output --DataDir ./dataset
```

This will process the data, train the Attention Residual UNet model, and save the model checkpoints in the `models` directory.


## Model predictions

The model predicts mask of structure and tumor as seen in images below:

### Liver Mask:
![alt text](image.png)
### Vessel Mask:
![alt text](image-1.png)

## Authors and acknowledgment

I would like to express my sincere gratitude to Prof. Dr.-Ing.
Klaus Drechsler and Tobias Holmes, M.Sc., for their invaluable
guidance and support throughout this project. Their insights
and expertise were instrumental in shaping the direction and
success of this work.

## License

This project is licensed under the MIT License.