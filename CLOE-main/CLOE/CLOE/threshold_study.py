import torch
import numpy as np
import math as m
import pickle as pkl
import matplotlib.pyplot as plt
import argparse
import time
import os
import importlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from christoffel import CLOE
from autoencoder_v2 import Autoencoder

import optuna
from sklearn.metrics import f1_score
import numpy as np
 
def objective(trial, scores, y_true):
    """
    Objective function for Optuna to optimize the threshold
    """
    # Define the threshold parameter to optimize
    threshold = trial.suggest_float('threshold', min(scores), max(scores))
    # Convert scores to binary predictions based on the threshold
    y_pred = [1 if score >= threshold else 0 for score in scores]
    # Calculate F1-score
    f1 = f1_score(y_true, y_pred)
    return f1
 
def find_best_threshold(scores, y_true):
    """
    Find the best threshold using Optuna optimization
    Args:
        scores: List of prediction scores/probabilities
        y_true: List of true binary labels (0 or 1)
    Returns:
        best_threshold: The threshold that gives the best F1-score
        best_f1_score: The best F1-score achieved
    """
    # Create study object to maximize F1-score
    study = optuna.create_study(direction='maximize')
    # Optimize the objective function
    study.optimize(lambda trial: objective(trial, scores, y_true), n_trials=100)
    # Get the best threshold and F1-score
    best_threshold = study.best_params['threshold']
    best_f1_score = study.best_value
    return best_threshold, best_f1_score

def print_v(args, string):
    """
        Print a message if verbosity is enabled in args.
    """
    if args.verbose :
        print(string)

def compute_threshold(pred):
    thresholds =  [torch.max(pred)]+ [torch.quantile(pred, 0.9)] +  [torch.quantile(pred, 0.75)] + [torch.quantile(pred, 0.5)] 
    return(thresholds)

def compute_scores(preds, threshold):
    return [1 if pred >= threshold else 0 for pred in preds]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=49,
                        help='seed')
    parser.add_argument('--n-support', type=int, default=None,
                        help='degree of the poynomial to compute the support for joint training step')
    parser.add_argument('--save-path', type=str, default='CLOE/datasets/models/',
                        help='Path to the folder to save results')
    parser.add_argument('--data-name', type=str, default='15_Hepatitis',
                        help='Name of the dataset, must be a numpy file in configs[\'data_dir\']')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
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
    n_support = args.n_support
    data_dir = model_config['data-dir']
    NUM_CLASSES = model_config['nb-class']
    type_conc = model_config['type-conc']
    lambda_CLOE = model_config['lambda-CLOE']
    DIM = model_config['dim']
    umap_enable = model_config['umap']

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
        test_size = 1- 5000/x[y==0].shape[0]
    X_train_valid, X_test= train_test_split(x[y==0], test_size=test_size, random_state=RANDOM_SEED)
    X_train, X_valid = train_test_split(X_train_valid, test_size=0.2, random_state=RANDOM_SEED)
    print_v(args,f'Train set shape: {X_train.shape}, valid set shape: {X_valid.shape} and test set shape: {X_test.shape}')

    # Load trained autoencoder
    autoencoder = Autoencoder(in_shape=x.shape[1], enc_shape=NUM_CLASSES, DIM = DIM, dropout_rate = 0).double().to(args.device)
    file_save = os.path.join(save_path,'jointrained.pt')
    autoencoder.load_state_dict(torch.load(file_save))
    print_v(args, 'Trained model loaded')
    print_v(args, autoencoder)

    # Computing the support
    autoencoder.eval()
    with torch.no_grad():
        x_encoded = autoencoder.encode(X_train.to(dtype=torch.float64))
        x_encoded_valid = autoencoder.encode(X_valid.to(dtype=torch.float64))
    christoffel_support = CLOE(n = n_support, regularization = "max", polynomial_basis = "monomials", inv = 'fpd_inv', device = args.device)
    christoffel_support.fit(x_encoded, x_valid=x_encoded_valid)
    print_v(args, 'Computing of the support with the model done')

    # Inference
    with torch.no_grad():
        x_encoded_test = autoencoder.encode(x.to(dtype=torch.float64))
    preds = []
    for x_reduce in torch.split(x_encoded_test,10000): # For very large dataset, spilt it and infere on each part.
        preds.append(christoffel_support.score_samples_noreg(x_reduce).detach())
    contamination = 1-sum(y)/len(y)

    best_threshold, best_f1 = find_best_threshold(np.concat(preds), y)
    ideal = np.percentile(np.concat(preds), contamination*100)
    train_based = compute_threshold(christoffel_support.score_samples_noreg(x_encoded).detach())
    valid_based = compute_threshold(christoffel_support.score_samples_noreg(x_encoded_valid).detach())

    aucroc = roc_auc_score(y_true=y, y_score=np.concat(preds))
    aucap = average_precision_score(y_true=y, y_score=np.concat(preds), pos_label=1)
    print(f'AUC ROC for Christoffel score: {aucroc}')
    print(f'AP AUC for Christoffel score: {aucap}')

    thresholds = [ideal] + train_based + valid_based + [best_threshold]
    name_threshold = ["ideal threshold", "max train threshold", "q90% threshold", "q75% threshold", "q50% threshold", "max valid threshold", "q90% valid threshold", "q75% valid threshold", "q50% valid threshold", "best threshold"]
    scores = [compute_scores(np.concat(preds), threshold) for threshold in thresholds]
    f1Score = [f1_score(y_true=y, y_pred=score) for score in scores]

    print(f'F1 Score for Christoffel score: {f1Score}')

    result_path = f"CLOE/results/{data_name}/{RANDOM_SEED}/CLOE_threshold_study/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "ideal threshold": f1Score[0],
                "max train threshold": f1Score[1],
                "q90% threshold": f1Score[2],
                "q75% threshold": f1Score[3],
                "q50% threshold": f1Score[4],
                "max valid threshold": f1Score[5],
                "q90% valid threshold": f1Score[6],
                "q75% valid threshold": f1Score[7],
                "q50% valid threshold": f1Score[8],
                "best threshold": f1Score[-1]
            },
        )
    if umap_enable:
        # Compute UMAP representation of the original data and displays outliers find by CLOE (in red)
        import umap
        perplexity = 30
        metric='euclidean'
        min_dist = 0.1

        umap_ = umap.UMAP(n_neighbors=perplexity, metric=metric, min_dist=min_dist,n_components=2)
        umap_.fit(x_encoded.numpy())
        X_embedded = umap_.transform(x_encoded_test.numpy())

        green_true = y == 0
        red_true = y == 1
        fig, ax = plt.subplots(1,len(scores)+1)
        ax[0].scatter(X_embedded[green_true, 0], X_embedded[green_true, 1], c="g", marker='x')
        ax[0].scatter(X_embedded[red_true, 0], X_embedded[red_true, 1], c="r", marker='x')
        ax[0].set_title('Ground truth')
        for i in range(len(scores)):
            green = [True if score == 0 else False for score in scores[i]]
            red = [True if score == 1 else False for score in scores[i]]

            ax[i+1].scatter(X_embedded[green, 0], X_embedded[green, 1], c="g", marker='x')
            ax[i+1].scatter(X_embedded[red, 0], X_embedded[red, 1], c="r", marker='x')
            
            ax[i+1].set_title(name_threshold[i])
        fig.set_figwidth(20)
        fig.set_figheight(7)
        plt.tight_layout() 
        plt.savefig(os.path.join(result_path,'umap_score.png'))
        plt.show()



