import torch
import numpy as np
import math as m
import pickle as pkl
import matplotlib.pyplot as plt
import argparse
import time
import os
import importlib

from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from christoffel import CLOE, ChristoffelScore_loss
from autoencoder_v2 import train, Autoencoder

def print_v(args, string):
    """
        Print a message if verbosity is enabled in args.
    """
    if args.verbose :
        print(string)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=49,
                        help='seed')
    parser.add_argument('--n', type=int, default=2,
                        help='degree of the poynomial to compute the support for pretraining step')
    parser.add_argument('--n-support', type=int, default=None,
                        help='degree of the poynomial to compute the support for joint training step')
    parser.add_argument('--save-path', type=str, default='CLOE/datasets/models/',
                        help='Path to the folder to save results')
    parser.add_argument('--data-name', type=str, default='15_Hepatitis',
                        help='Name of the dataset, must be a numpy file in configs[\'data_dir\']')
    parser.add_argument('-v', '--verbose', type=bool, default=True,
                        help='Print the log in console')
    
    args = parser.parse_args()
    dict_to_import = 'model_config_CLOE'
    module_name = 'configs'
    module = importlib.import_module(module_name)
    model_config = getattr(module, dict_to_import)

    RANDOM_SEED = args.seed

    # Set the random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)

    # Choose CPU or GPU if available automatically
    if torch.cuda.is_available():
        print_v(args, 'Using Cuda')
        torch.cuda.manual_seed(RANDOM_SEED)
        args.device = 'cuda:0'
    else : 
        args.device = 'cpu'

    time_begin = time.time()

    # Hyperparameters
    n = args.n
    n_support = args.n_support
    lr = model_config['learning-rate']
    data_dir = model_config['data-dir']
    NUM_EPOCHS_PRE = model_config['epochs-pre']
    NUM_EPOCHS_JOINT = model_config['epochs-joint']
    NUM_CLASSES = model_config['nb-class']
    type_conc = model_config['type-conc']
    lambda_CLOE = model_config['lambda-CLOE']
    DIM = model_config['dim']
    patience = model_config['patience']
    umap_enable = model_config['umap']

    # Enable multiprocessing
    NUM_WORKER = model_config['num-workers']

    # Dataset preprocessing
    data_name = args.data_name
    save_path = os.path.join(args.save_path, data_name)
    data = np.load(os.path.join(data_dir,args.data_name+'.npz'), allow_pickle=True)
    X, y = data['X'], data['y']
    if data_name == "24_mnist":
        x = torch.from_numpy(RobustScaler().fit_transform(X)).to(args.device)
    else:
        x = torch.from_numpy(StandardScaler().fit_transform(X)).to(args.device)
    print_v(args, f'Number of instances : {X.shape[0]}, number of dimension: {X.shape[1]}')

    if data_name == "9_census":
         test_size = 1- 3000/x[y==0].shape[0]
    elif data_name == "24_mnist":
         test_size = 1- 5000/x[y==0].shape[0]
    elif x[y==0].shape[0]<8000:
        test_size = 0.1
    else:
        test_size = 1- 8000/x[y==0].shape[0]
    X_train_valid, X_test= train_test_split(x[y==0], test_size=test_size, random_state=RANDOM_SEED)
    X_train, X_valid = train_test_split(X_train_valid, test_size=0.2, random_state=RANDOM_SEED)
    print_v(args,f'Train set shape: {X_train.shape}, valid set shape: {X_valid.shape} and test set shape: {X_test.shape}')

    # Compute automatically the batch size depending of n and N
    BATCH_SIZE = m.comb(NUM_CLASSES+n, n)
    print_v(args, f'Batch size is: {BATCH_SIZE}')

    # Pre training
    dropout_rate = 0.2
    autoencoder = Autoencoder(in_shape=x.shape[1], enc_shape=NUM_CLASSES, DIM = DIM, dropout_rate = dropout_rate).double().to(args.device)
    print_v(args, autoencoder)
    error = ChristoffelScore_loss(n=n, type_conc=type_conc, step_training=0, random_seed = RANDOM_SEED, device = args.device, lambda_CLOE = lambda_CLOE) 

    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    # torch.autograd.set_detect_anomaly(True)

    file_save = os.path.join(save_path,'pretrain.pt')
    # if not os.path.isfile(file_save):
    memory_list = train(
            model              = autoencoder,
            train_loader       = DataLoader(X_train.to(dtype=torch.float64), batch_size = BATCH_SIZE, num_workers = NUM_WORKER),
            validation_loader  = DataLoader(X_valid.to(dtype=torch.float64)),
            epochs             = NUM_EPOCHS_PRE,
            device             = args.device,
            optimizer          = optimizer, 
            loss_function      = error, 
            patience_max       = patience,
            train_mode         = 0,
            path_save          = save_path)
    print_v(args, 'Pre-training of the model done')

    # Joint training 
    dropout_rate = 0
    autoencoder = Autoencoder(in_shape=x.shape[1], enc_shape=NUM_CLASSES, DIM = DIM, dropout_rate = dropout_rate).double().to(args.device)
    autoencoder.load_state_dict(torch.load(file_save))
    print_v(args, 'pre-trained model loaded')
    print_v(args, autoencoder)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    error = ChristoffelScore_loss(n=n, type_conc=type_conc, step_training=1, random_seed = RANDOM_SEED, device = args.device, lambda_CLOE = lambda_CLOE) 
    if data_name == "3_backdoor" :
        batch_size = m.comb(NUM_CLASSES+n, n)
    else:
        batch_size = min(X_train.shape[0], 1500)
    print_v(args, f'batch size for joint training step = {batch_size}')
    memory_list = train(
            model              = autoencoder,
            train_loader       = DataLoader(X_train.to(dtype=torch.float64), batch_size = batch_size, num_workers = NUM_WORKER, drop_last=True),
            validation_loader  = DataLoader(X_valid.to(dtype=torch.float64)),
            epochs             = NUM_EPOCHS_JOINT,
            device             = args.device,
            optimizer          = optimizer, 
            loss_function      = error,
            patience_max       = patience,
            train_mode         = 1,
            path_save          = save_path)
    print_v(args, 'Joint-training of the model done')
    
    # Support computing
    file_save =  os.path.join(save_path,'jointrained.pt')
    autoencoder.load_state_dict(torch.load(file_save))
    print_v(args, 'joint-trained model loaded')
    autoencoder.eval()
    with torch.no_grad():
        x_encoded = autoencoder.encode(X_train.to(dtype=torch.float64))
        x_encoded_valid = autoencoder.encode(X_valid.to(dtype=torch.float64))
    christoffel_support = CLOE(n = n_support, regularization = "max", polynomial_basis = "monomials", inv = 'fpd_inv', device = args.device)
    christoffel_support.fit(x_encoded, x_valid=x_encoded_valid)
    print_v(args, 'Computing of the support with the model done')
    time_training = time.time()-time_begin
    print(f'Time train: {time_training}')

    # Inference
    
    with torch.no_grad():
        x_encoded_test = autoencoder.encode(x.to(dtype=torch.float64))
    pred = []
    score = []
    for x_reduce in torch.split(x_encoded_test,10000): # For very large dataset, spilt it and infere on each part.
        pred.append(christoffel_support.score_samples_noreg(x_reduce).detach())
    time_begin_inf = time.time() 
    for x_reduce in torch.split(x_encoded_test,10000): # For very large dataset, spilt it and infere on each part.
        score.append(christoffel_support.predict(x_reduce).detach())
    aucroc = roc_auc_score(y_true=y, y_score=np.concat(pred))
    aucap = average_precision_score(y_true=y, y_score=np.concat(pred), pos_label=1)
    time_inference = time.time()-time_begin_inf
    f1Score = f1_score(y_true=y, y_pred=np.concat(score))
    print(f'AUC ROC for Christoffel score: {aucroc}')
    print(f'AP AUC for Christoffel score: {aucap}')
    print(f'F1 Score for Christoffel score: {f1Score}')
    print(f'Time inference: {time_inference}')
    print(f'Time inference for one sample: {time_inference/x_encoded_test.shape[0]}')
    result_path = f"CLOE/results/{data_name}/{RANDOM_SEED}/CLOE/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "AUC ROC": aucroc,
                "AP AUC": aucap,
                "F1 Score": f1Score,
                "Time training": time_training,
                "Time inference": time_inference,
                "Time inference for one sample": time_inference/x_encoded_test.shape[0],
                "n": christoffel_support.n,
                "mean_memory": sum(memory_list)/len(memory_list)
            },
        )
    if umap_enable:
        # Compute UMAP representation of the original data and displays outliers find by CLOE (in red)
        import umap
        perplexity = 30
        metric='euclidean'
        min_dist = 0.1

        umap_ = umap.UMAP(random_state=RANDOM_SEED, n_neighbors=perplexity, metric=metric, min_dist=min_dist,n_components=2)
        umap_.fit(X)
        X_embedded = umap_.transform(X)

        green_true = y == 0
        red_true = y == 1
        fig, ax = plt.subplots(1,2)
        ax[0].scatter(X_embedded[green_true, 0], X_embedded[green_true, 1], c="g", marker='x')
        ax[0].scatter(X_embedded[red_true, 0], X_embedded[red_true, 1], c="r", marker='x')
        ax[0].set_title('Ground truth')
        
        green = np.concat(score) == 0
        red = np.concat(score) == 1
        
        ax[1].scatter(X_embedded[green, 0], X_embedded[green, 1], c="g", marker='x')
        ax[1].scatter(X_embedded[red, 0], X_embedded[red, 1], c="r", marker='x')
        
        ax[1].set_title("CLOE")
        fig.set_figwidth(15)
        fig.set_figheight(7)
        plt.savefig(os.path.join(save_path,'umap_score.png'))



