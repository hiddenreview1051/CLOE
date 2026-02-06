# Code for the 4 ablation experiments
import os
import torch
from torch import nn
import numpy as np
import math as m
import pickle as pkl
import matplotlib.pyplot as plt
import argparse

from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from christoffel import CLOE, ChristoffelScore_loss
from autoencoder_v2 import train, Autoencoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from pyod.models.kde import KDE
from torchkde import KernelDensity
from pyod.models.ocsvm import OCSVM

def test(christoffel_model, x_enc, y, args):
    pred = christoffel_model.score_samples_noreg(x_enc).detach()
    aucroc = roc_auc_score(y_true=y, y_score=pred)
    aucap = average_precision_score(y_true=y, y_score=pred, pos_label=1)
    score = christoffel_model.predict(x_enc).detach()
    f1Score = f1_score(y_true=y, y_pred=score)

    print(f'AU-ROC for Christoffel score: {aucroc}')
    print(f'AP AUC for Christoffel score: {aucap}')
    print(f'F1 Score for Christoffel score: {f1Score}')

    result_path = f"CLOE/results/{args.data_name}/{args.seed}/CLOE_ablation_{args.study}/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "AUC ROC": aucroc,
                "AP AUC": aucap,
                "F1 Score": f1Score,
            },
        )
def test_kde(model, x_enc, y, args):
    pred = -model.score_samples(x_enc)
    aucroc = roc_auc_score(y_true=y, y_score=pred)
    aucap = average_precision_score(y_true=y, y_score=pred, pos_label=1)

    print(f'AU-ROC for Christoffel score: {aucroc}')
    print(f'AP AUC for Christoffel score: {aucap}')

    result_path = f"CLOE/results/{args.data_name}/{args.seed}/CLOE_ablation_{args.study}/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "AUC ROC": aucroc,
                "AP AUC": aucap,
            },
        )


class KDELoss(nn.Module):
    def __init__(self, contamination, step_training: int, random_seed:int = 49, device:str ='cpu'):
        super().__init__()
        self.step_training = step_training
        self.random_seed = random_seed
        self.MSE = nn.MSELoss()
        self.device = device
        self.contamination = contamination

    
    def forward(self, output, x, x_enc, dataset, valid = False):
        mse_loss = self.MSE(output, x)
        if self.step_training == 0 : 
            return mse_loss, mse_loss, torch.zeros(1)
        else : 
            if not x_enc.requires_grad:
                    x_enc = x_enc.clone().detach().requires_grad_(True)
            if valid :
                X_support = dataset # compute the support with all training data
            else : 
                batch_size = x_enc.shape[0]
                indices = torch.randperm(batch_size, device=x_enc.device)
                split_idx = int(batch_size * 0.8)
                support_idx = indices[:split_idx]
                X_support = x_enc[support_idx] 
            score = -KernelDensity(bandwidth=.4, kernel='gaussian').fit(X_support).score_samples(x_enc)
            return  mse_loss + torch.mean(score),  mse_loss ,  torch.mean(score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=49,
                        help='seed')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--n', type=int, default=2,
                        help='degree of the poynomial to compute the support')
    parser.add_argument('--study', type=int, default=0,
                        help='0 for removing the joint training, 1 for removing the pre training, 2 for no training at all, 3 for KDE')
    parser.add_argument('--nb-epochs', type=int, default=10,
                        help='Number of epochs for this training step')
    parser.add_argument('--nb-class', type=int, default=8,
                        help='Dimension of the latent space of the autoencoder')
    parser.add_argument('--dataset-path', type=str, default='CLOE/datasets/',
                        help='Path to the dataset (numpy file)')
    parser.add_argument('--data-name', type=str, default='15_Hepatitis',
                        help='Name of the dataset (numpy file)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping')
    parser.add_argument('--lambda_CLOE', type=int, default=1,
                        help='coefficient lambda in the training loss for the Christoffel function part')
    parser.add_argument('--dim', type=int, default=[500, 500, 2000], nargs='+',
                        help='Dimension of the hidden layer of the encoder in the order')
    parser.add_argument('--type-conc', type=str, default='mean',
                        help='type of the concatenation for all the Christoffel value in the loss : mean, sum or max')
    parser.add_argument('--num-worker', type=int, default=0,
                        help='Number of worker used to train the model')
    parser.add_argument('--umap', type=bool, default=False,
                        help='Save the image of the UMAP representation of the data with inliers in green and outliers in red')


    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Choose CPU or GPU if available automatically
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        args.device = 'cuda:0'
    else : 
        args.device = 'cpu'

    print(f"Type ablation study: {args.study}")

    # Hyperparameters
    RANDOM_SEED = args.seed
    LEARNING_RATE = args.lr
    n = args.n
    NUM_EPOCHS = args.nb_epochs
    NUM_CLASSES = args.nb_class
    
    study = args.study
    type_conc = args.type_conc
    lambda_CLOE = args.lambda_CLOE
    DIM = args.dim

    # Enable multiprocessing
    NUM_WORKER = args.num_worker
    if NUM_WORKER > 1 :
        torch.set_num_threads(NUM_WORKER) 
        torch.set_num_interop_threads(NUM_WORKER) 

    # Dataset preprocessing
    data_name = args.data_name
    data = np.load(f'{args.dataset_path}{args.data_name}.npz', allow_pickle=True)
    X, y = data['X'], data['y']
    x = torch.from_numpy(StandardScaler().fit_transform(X)).to(args.device)

    contamination = len(y[y==1])/len(y)

    # Compute automatically the batch size depending of n and N
    BATCH_SIZE = m.comb(NUM_CLASSES+n, n)

    train_mode = {
        "pre-training": 0,
        "joint-training": 1,
        "compute-support": 2
    }

    if args.study == 0 or args.study == 2  or args.study == 3 :
        dropout_rate = 0.2
        training_step = "pre-training"
        file_save = f'CLOE/models_abl_{args.study}/{data_name}_{type_conc}_{int(lambda_CLOE)}/pretrain'

    elif args.study == 1:
        dropout_rate = 0.0
        training_step = "joint-training"
        file_save = f'CLOE/models_abl_{args.study}/{data_name}_{type_conc}_{int(lambda_CLOE)}/jointrained'
    
    autoencoder = Autoencoder(in_shape=x.shape[1], enc_shape=NUM_CLASSES, DIM = DIM, dropout_rate = dropout_rate).double().to(args.device)
    
    if args.study == 0 or args.study == 1 or args.study == 2:
        error = ChristoffelScore_loss(n=n, type_conc=type_conc, step_training=train_mode[training_step], random_seed = RANDOM_SEED)
    elif args.study == 3 : 
        error = KDELoss(step_training=train_mode[training_step], random_seed = RANDOM_SEED, contamination=contamination)

    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    torch.autograd.set_detect_anomaly(True)

    x_success = x[y==0]
    if data_name == "9_census":
        test_size = 1- 3000/x[y==0].shape[0]
    elif data_name == "24_mnist":
        test_size = 1- 5000/x[y==0].shape[0]
    elif x_success.shape[0]<5000:
        test_size = 0.1
    else:
        test_size = 1- 5000/x_success.shape[0]
    X_train_valid, X_test= train_test_split(x_success, test_size=test_size, random_state=RANDOM_SEED)
    X_train, X_valid = train_test_split(X_train_valid,test_size=0.2, random_state=RANDOM_SEED)

    if args.study != 2:
        train(
            model              = autoencoder,
            train_loader       = DataLoader(X_train.to(dtype=torch.float64), batch_size = BATCH_SIZE, num_workers = NUM_WORKER),
            validation_loader  = DataLoader(X_valid.to(dtype=torch.float64)),
            epochs             = NUM_EPOCHS,
            device             = args.device,
            optimizer          = optimizer, 
            loss_function      = error,
            patience_max       = 20,
            train_mode         = train_mode[training_step],
            path_save          = f'CLOE/models_abl_{args.study}/{data_name}_{type_conc}_{int(lambda_CLOE)}')
        autoencoder.load_state_dict(torch.load( f'{file_save}.pt'))

    if args.study == 3 :
        dropout_rate = 0.0
        training_step = "joint-training"
        error = KDELoss(step_training=train_mode[training_step], random_seed = RANDOM_SEED, contamination=contamination)
        file_save = f'CLOE/models_abl_{args.study}/{data_name}_{type_conc}_{int(lambda_CLOE)}/jointrained'
        autoencoder = Autoencoder(in_shape=x.shape[1], enc_shape=NUM_CLASSES, DIM = DIM, dropout_rate = dropout_rate).double().to(args.device)
        optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
        train(
            model              = autoencoder,
            train_loader       = DataLoader(X_train.to(dtype=torch.float64), batch_size = BATCH_SIZE, num_workers = NUM_WORKER),
            validation_loader  = DataLoader(X_valid.to(dtype=torch.float64)),
            epochs             = 150,
            device             = args.device,
            optimizer          = optimizer, 
            loss_function      = error,
            patience_max       = 20,
            train_mode         = train_mode[training_step],
            path_save          = f'CLOE/models_abl_{args.study}/{data_name}_{type_conc}_{int(lambda_CLOE)}',
            abl3               = True)
        
        autoencoder.load_state_dict(torch.load( f'{file_save}.pt'))

    # Compute the support
    with torch.no_grad():
        autoencoder.eval()
        x_encoded = autoencoder.encode(x)
    
    X_train_valid, X_test = train_test_split(x_encoded[y==0], test_size=test_size, random_state=RANDOM_SEED)
    X_train, X_valid= train_test_split(X_train_valid, test_size=0.2, random_state=RANDOM_SEED)

    if args.study == 0 or args.study == 1 or args.study == 2:
        christoffel_support = CLOE(n=None, regularization= "max", polynomial_basis = "monomials", inv = 'fpd_inv')
        christoffel_support.fit(X_train, X_valid)
        # Compute the metrics
        test(christoffel_support, x_encoded, y, args)
    elif args.study == 3 : 
        model = KernelDensity(bandwidth=.4, kernel='gaussian').fit(X_train)
        test_kde(model, x_encoded, y, args)

    
            