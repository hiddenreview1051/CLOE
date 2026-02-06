# CLOE: Christoffel Loss Autoencoder for Anomaly Detection

This project implements the code from the article : CLOE: Christoffel LOss autoEncoder for anomaly detection

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

## Installation

Python 3.13 is required to run the project.  
To run DAGMM baseline experiments, Python 3.9 or lower is required, uncomment the package adbench in file 'requirements.txt'.  

Then install required packages with the file 'requirements.txt':  
`pip install -r requirements.txt`
## Usage
For all the CLOE different training, `--verbose True` can be added to the command line to enable verbose during training and inference.  

The required dataset are stored in `CLOE/datasets`.  

Results of the metrics are displayed in the terminal and save in CLOE/results. 

## All expermients in the paper
To run all paper experiments, except DAGMM and get results, launch the following command:

`run.sh`

### Full training and inference for one dataset
To launch a training for the three training steps and the inference use this command (example for Hepatitis dataset):  

`python3.13 CLOE/CLOE/main.py --data-name 15_Hepatitis --n 2`  

Attributes:  
```
    n:int
        The degree of the polynomials of the Christoffel Function for the joint training step, set between 2 and 6,
    n-support:int
        The degree of the polynomials of the Christoffel Function for the support computing step, set between 2 and 6, if None automatically computed.
    save-path: str
        Path to the folder to save results,
    data-name: str
        Name of the dataset, must be a numpy file in configs['data_dir'],
    verbose: bool
        If the log must be printed in the terminal.
```  
All the other parameters of training can be configured in the file: CLOE/CLOE/configs.py  
```   
    'epochs_pre':int
        Number of epochs for the pre-training step (default is 10)
    'epochs_joint':int
        Number of epochs for the joint-training step (default is 150)
    'learning_rate':float
        Value of the learning rate (default is 1e-4)
    'data_dir':str
        Path to the data folder (default is 'CLOE/datasets/')
    'dim':[int]
        Dimension of the hidden layers of the autoencoder (default is [500, 500, 2000]),
    'random_seed':int
        Random seed to reproduce results (default is 49),
    'num_workers':int
        Number of workers (default is 0),
    'patience':int
        Number of epochs to wait before stop training if the validation loss does not decrease (default is 20),
    'nb-class':int 
        Dimension of the latent space (default is 8),
    'type-conc':str
        Type of the concatenation of the two loss functions: choice between 'mean', 'max' and 'sum' (default is 'mean'),
    'umap':bool
        Save the UMAP 2D projection of the original data with the outliers (red) and inliers (green) detected by the method (default is False).
```  
### Pre training
To launch only the training step :  
`python3.13 CLOE/CLOE/train.py --n 2 --nb-epochs 10 --data-name 15_Hepatitis --training-step pre-training`

### Joint training
To launch only the joint training step (a pretraining step must have been executed before):  
`python3.13 CLOE/CLOE/train.py --n 2 --nb-epochs 150 --data-name 15_Hepatitis --training-step joint-training`

### Support computing and inference
To launch only the computing of the support (a joint training step must have been executed before):  
`python3.13 CLOE/CLOE/train.py --n 3 --data-name 15_Hepatitis --training-step compute-support`

### Inference
To infer a training model on a dataset:
`python3.13 CLOE/CLOE/test.py --n 3 --data-name 15_Hepatitis --model-path CLOE/datasets/models/15_Hepatitis`

### Baselines
For OC-SVM, iForest, KNN, ECOD and DeepSVDD:   
`python3.13 CLOE/baseline/train_test.py --data-name 15_Hepatitis --oc-svm True --iforest True --knn True --kde True --ecod True --deepSVDD True`  
For DAGMM (install package adbench is resquired with Python3.9), all datasets in the folder `CLOE/datasets` will be used:  
`python3.9 CLOE/baseline/dagmm.py `   
For DRL:  
`python3.13 CLOE/baseline/DRL/main.py --dataname 15_Hepatitis --model_type DRL --preprocess standard --diversity True --plearn False --input_info True --input_info_ratio 0.1 --cl True --cl_ratio 0.06 --basis_vector_num 5 --seed 49`  
For RCA:
`python3.13 CLOE/baseline/RCA/trainRCA.py --data 15_Hepatitis --seed 49 --training_ratio 0.599 --max_epochs 200 --hidden_dim 128 --z_dim 10`
For MCM:
`python3.13 CLOE/baseline/MCM/main.py --data-name 15_Hepatitis --seed 49`     
For DDAE:  
`python3.13 CLOE/baseline/DRL/main.py -- --data-name 15_Hepatitis --seed 49`
            
## Configuration

### Available datasets
| Name        | in command line | n | n-support |
| ----------- | --------------- |---|-----------|
| ALOI        | 1_ALOI          | 5 | 6         |
| backdoor    | 3_backdoor      | 5 | 6         |
| breastw     | 4_breastw       | 4 | 5         |
| campaign    | 5_campaign      | 5 | 6         |
| cardio      | 6_cardio        | 4 | 6         |
| census      | 9_census        | 5 | 5         |
| fault       | 12_fault        | 4 | 5         |
| Hepatitis   | 15_Hepatitis    | 2 | 3         |
| InternetAds | 17_InternetAds  | 4 | 5         |
| landsat     | 19_landsat      | 4 | 6         |
| letter      | 20_letter       | 4 | 5         |
| mnist       | 25_mnist        | 4 | 6         |
| musk        | 25_musk         | 4 | 6         |
| shuttle     | 32_shuttle      | 5 | 6         |
| speech      | 36_speech       | 4 | 6         |




