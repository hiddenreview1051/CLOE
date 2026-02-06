# Need Python 3.9 or less to execute this code

from  adbench.baseline.DAGMM.train import TrainerDAGMM
from  adbench.baseline.DAGMM.test import eval
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse 
import torch

parser = argparse.ArgumentParser()

args = parser.parse_args()


args.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.n_gmm = 5
args.latent_dim = 1
args.num_epochs = 200
args.patience = 50
args.lr = 1e-4
args.lambda_energy = 0.1
args.lambda_cov = 0.005

path = "CLOE/datasets"

for file in os.listdir(path):
    data_name = file[:-4]
    print(data_name)
    data = np.load(os.path.join(path, file), allow_pickle=True)
    X, y = data['X'], data['y']

    if X[y==0].shape[0]<5000:
        test_size = 0.1
    else:
        test_size = 1- 5000/X[y==0].shape[0]
    X_train_valid, X_test = train_test_split(X[y==0].to(dtype=torch.float64).cpu().numpy(), test_size=test_size, random_state=49)
    X_train, X_valid= train_test_split(X_train_valid, test_size=0.2, random_state=49)
    args.batch_size = X_train.shape[0]

    data_test = {
        'X_train' : X_train,
        'X_test' : X
        }

    print(f'test size: {X_test.shape}')
    print(f'train size: {X_train.shape}')
    print(f'valid size: {X_valid.shape}')
    
    dagmm_trainer = TrainerDAGMM(args, X_train, args.device)
    dagmm_trainer.train()
    energy_test = eval(dagmm_trainer.model, data_test, args.device, n_gmm=5, batch_size = args.batch_size)

    print(energy_test)

    roc_auc = roc_auc_score(y_true=y, y_score=energy_test)
    ap = average_precision_score(y_true=y, y_score=energy_test)

    print(f'AU ROC: {roc_auc}')
    print(f'AP: {ap}')
    result_path = f"CLOE/results/{args.data_name}/{args.seed}/dagmm/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "AUC ROC": roc_auc,
                "AP AUC": ap,
                "F1 Score": 0,
            },
        )
