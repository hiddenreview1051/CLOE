import numpy as np
import math as m
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import os

from autoencoder_v2 import Autoencoder
from christoffel import CLOE

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=49,
                    help='seed')
parser.add_argument('--n', type=int, default=2,
                    help='degree of the poynomial to compute the support')
parser.add_argument('--dataset-path', type=str, default='CLOE/datasets/',
                    help='Path to the dataset (numpy file)')
parser.add_argument('--data-name', type=str, default='15_Hepatitis',
                    help='Name of the dataset to save the training model')
parser.add_argument('--model-path', type=str, default='CLOE/datasets/models/15_Hepatitis',
                    help='Path to the fully training model (pickle file)')
parser.add_argument('--num-worker', type=int, default=0,
                    help='Number of worker used to train the model')
parser.add_argument('--nb-class', type=int, default=8,
                    help='Dimension of the latent space of the autoencoder')
parser.add_argument('--dim', type=int, default=[500, 500, 2000], nargs='+',
                    help='Dimension of the hidden layer of the encoder in the order')
parser.add_argument('--umap', type=bool, default=False,
                    help='Save the image of the UMAP representation of the data with inliers in green and outliers in red')


args = parser.parse_args()

torch.manual_seed(args.seed)

#Choose CPU or GPU if available automatically
if torch.cuda.is_available():
    print('Using Cuda')
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
else : 
    args.device = 'cpu'


# Hyperparameters
RANDOM_SEED = args.seed
n = args.n
NUM_CLASSES = args.nb_class
DIM = args.dim


# Enable multiprocessing
NUM_WORKER = args.num_worker
if NUM_WORKER > 1 :
    torch.set_num_threads(NUM_WORKER) 
    torch.set_num_interop_threads(NUM_WORKER) 

# Dataset preprocessing
data_name = args.data_name
data = np.load(os.path.join(args.dataset_path, args.data_name+'.npz'), allow_pickle=True)
X, y = data['X'], data['y']
x = torch.from_numpy(StandardScaler().fit_transform(X)).to(args.device)
print(args, f'Number of instances : {X.shape[0]}, number of dimension: {X.shape[1]}')
BATCH_SIZE = m.comb(NUM_CLASSES+n, n)


autoencoder = Autoencoder(in_shape=x.shape[1], enc_shape=NUM_CLASSES, DIM = DIM, dropout_rate = 0).double().to(args.device)
file_autoencoder = os.path.join(args.model_path, 'jointrained.pt')
autoencoder.load_state_dict(torch.load(file_autoencoder))
autoencoder.eval()
with torch.no_grad():
        x_encoded = autoencoder.encode(x.to(dtype=torch.float64))

if data_name == "9_census":
        test_size = 1- 3000/x[y==0].shape[0]
elif data_name == "24_mnist":
        test_size = 1- 5000/x[y==0].shape[0]
elif x[y==0].shape[0]<8000:
    test_size = 0.1
else:
    test_size = 1- 8000/x[y==0].shape[0]
X_train_valid, X_test= train_test_split(x_encoded[y==0], test_size=test_size, random_state=RANDOM_SEED)
X_train, X_valid = train_test_split(X_train_valid, test_size=0.2, random_state=RANDOM_SEED)

christoffel_support = CLOE(n=n, regularization= "max", polynomial_basis = "monomials", inv = 'fpd_inv', device = args.device)
christoffel_support.fit(X_train)
score = christoffel_support.score_samples_noreg(x_encoded)
pred = christoffel_support.predict(x_encoded)
aucroc = roc_auc_score(y_true=y, y_score=score.detach().cpu())
aucap = average_precision_score(y_true=y, y_score=score.detach().cpu(), pos_label=1)
f1Score = f1_score(y_true=y, y_pred=pred.detach().cpu())

print(f'AUC ROC for Christoffel score: {aucroc}')
print(f'AP AUC for Christoffel score: {aucap}')
print(f'F1 Score for Christoffel score: {f1Score}')

if args.umap:
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
    
    green = pred == 0
    red = pred == 1
    
    ax[1].scatter(X_embedded[green, 0], X_embedded[green, 1], c="g", marker='x')
    ax[1].scatter(X_embedded[red, 0], X_embedded[red, 1], c="r", marker='x')
    
    ax[1].set_title("CLOE")
    fig.set_figwidth(15)
    fig.set_figheight(7)
    plt.savefig(os.path.join(args.model_path,'umap_score.png'))