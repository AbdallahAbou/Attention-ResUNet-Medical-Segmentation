# Python ML Template structure

## Overview

```bash
.
├── data                
│   ├── interim               <- intermediate data that has been transformed
│   ├── processed             <- the final dataset for model training
│   ├── raw                   <- the original data
│   └── results               <- data after training the model, i.e. weights
├── docs                      <- documentation for your project (e.g. markdown files)
│   └── data_structure.md     <- this file ;-)
├── .gitignore                <- ignore files that should not be commited to git
├── models                    <- trained models, model predictions or model summaries
├── notebooks                 <-  Jupyter notebooks. E.g. 1-th-data-exploration.ipynb 
├── README.md                 <- REAME file for this git repo
├── requirements.txt          <- pip modules to be installed. See "pip install -r"
└── src                       <- source code for this project
    ├── __init__.py           <- makes src a python module 
    ├── data                  <- scripts to download or generate data
    ├── models                <- scripts to train models and make predictions
    │   ├── train_model.py   
    │   └── predict_model.py 
    └── visualisation
        └──visualise.py       <- scripts to create visualisations
```