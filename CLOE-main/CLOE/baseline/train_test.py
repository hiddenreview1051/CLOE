import numpy as np
import math as m
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pickle as pkl
import torch
import argparse
import os


# Import the model
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.knn import KNN
from pyod.models.kde import KDE

def print_v(args, string):
    """
        Print a message if verbosity is enabled in args.
    """
    if args.verbose :
        print(string)

def save_result(method, aucroc, aucap, f1Score, args):
    result_path = f"CLOE/results/{args.data_name}/{args.seed}/{method}/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "AUC ROC": aucroc,
                "AP AUC": aucap,
                "F1 Score": f1Score,
            },
        )


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=49,
                    help='seed')
parser.add_argument('--dataset-path', type=str, default='CLOE/datasets/',
                    help='Path to the dataset (numpy file)')
parser.add_argument('--data-name', type=str, default='15_Hepatitis',
                    help='Name of the dataset to save the training model')
parser.add_argument('--num-worker', type=int, default=0,
                    help='Number of worker used to train the model')
parser.add_argument('--iforest', type=bool, default=False,
                    help='Test with iForest')
parser.add_argument('--knn', type=bool, default=False,
                    help='Test with KNN')
parser.add_argument('--kde', type=bool, default=False,
                    help='Test with KDE')
parser.add_argument('--ecod', type=bool, default=False,
                    help='Test with ECOD')
parser.add_argument('--oc-svm', type=bool, default=False,
                    help='Test with OC-SVM')
parser.add_argument('--deepSVDD', type=bool, default=False,
                    help='Test with DeepSVDD')
parser.add_argument('--save-path', type=str, default='CLOE/datasets/models/',
                    help='Path to the repo of the model to save the umap representation')
parser.add_argument('--umap', type=bool, default=False,
                    help='Save the image of the UMAP representation of the data with inliers in green and outliers in red')


args = parser.parse_args()

torch.manual_seed(args.seed)

#Choose CPU or GPU if available automatically
if torch.cuda.is_available():
    print(args, 'Using Cuda')
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
else : 
    args.device = 'cpu'

# Hyperparameters
RANDOM_SEED = args.seed

# Enable multiprocessing
NUM_WORKER = args.num_worker
if NUM_WORKER > 1 :
    torch.set_num_threads(NUM_WORKER) 
    torch.set_num_interop_threads(NUM_WORKER) 

# Dataset preprocessing
data_name = args.data_name
data = np.load(f'{args.dataset_path}{args.data_name}.npz', allow_pickle=True)
X, y = data['X'], data['y']
x = torch.from_numpy(X) #.to(args.device)
print(args, f'Number of instances : {X.shape[0]}, number of dimension: {X.shape[1]}')

if data_name == "9_census":
    test_size = 1- 3000/x[y==0].shape[0]
elif data_name == "24_mnist":
    test_size = 1- 5000/x[y==0].shape[0]
elif x[y==0].shape[0]<8000:
    test_size = 0.1
else:
    test_size = 1- 1500/x[y==0].shape[0]
# X_train_valid, X_test = train_test_split(x[y==0].to(dtype=torch.float64).cpu().numpy(), test_size=test_size, random_state=RANDOM_SEED)
X_train_valid, X_test = train_test_split(x[y==0], test_size=test_size, random_state=RANDOM_SEED)
X_train, X_valid= train_test_split(X_train_valid, test_size=0.2, random_state=RANDOM_SEED)

y_fail = y[y==1]
contamination = len(y_fail)/len(y)
print(f'contamination: {contamination}')

y_list = []
methods= []
y_pred = []

save_path = os.path.join(args.save_path, data_name)

if args.ecod :

    def ecod (X, contamination):
        # train ECOD detector
        clf = ECOD()
        clf = ECOD(contamination=contamination, n_jobs=args.num_worker)
        clf.fit(X)
        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        return clf, y_train_pred

    def ecod_test (clf, X_test):
        return  clf.predict(X_test), clf.decision_function(X_test)
    
    methods.append('ECOD')
    clf, y_train_pred = ecod(X_train, contamination=contamination)
    y_ecod, y_test_scores = ecod_test(clf, x.cpu().numpy())
    y_list.append(y_ecod)
    y_pred.append(y_test_scores)

if args.iforest :
    methods.append('iForest')
    iforest = IForest(contamination=contamination)
    iforest.fit(X_train)
    y_iforest = iforest.predict(x.cpu().numpy())
    y_list.append(y_iforest)
    y_pred.append(iforest.decision_function(x.cpu().numpy()))

if args.knn :
    methods.append('kNN')
    knn = KNN(contamination=contamination)
    knn.fit(X_train)
    y_knn = knn.predict(x.cpu().numpy())
    y_list.append(y_knn)
    y_pred.append(knn.decision_function(x.cpu().numpy()))

if args.kde :
    from torchkde import KernelDensity
    methods.append('KDE')
    kde = KDE(contamination=contamination)
    kde.fit(X_train)
    y_kde = kde.predict(x.cpu().numpy())
    y_list.append(y_kde)
    y_pred.append(kde.decision_function(x.cpu().numpy()))
    # kde = KernelDensity(bandwidth=.20, kernel='gaussian').fit(X_train)
    # y_list.append(torch.nan_to_num(-kde.score_samples(x), nan = 0.0, posinf=1000, neginf=-1000))
    # y_pred.append( torch.nan_to_num(-kde.score_samples(x), nan = 0.0, posinf=1000, neginf=-1000))

if args.oc_svm :
    methods.append('OC SVM') 
    ocsvm = OCSVM()
    ocsvm.fit(X_train)
    y_ocsvm = ocsvm.predict(x.cpu().numpy())
    y_list.append(y_ocsvm)
    y_pred.append(ocsvm.decision_function(x.cpu().numpy()))

if args.deepSVDD :
    methods.append('DeepSVDD')
    deep_svdd = DeepSVDD(n_features = x.shape[1])
    deep_svdd.fit(X_train)
    y_deep_svdd = deep_svdd.predict(x.cpu().numpy())
    y_list.append(y_deep_svdd)
    y_pred.append(deep_svdd.decision_function(x.cpu().numpy()))

for i, score in enumerate (y_list) : 
    aucroc = roc_auc_score(y_true=y, y_score=y_pred[i])
    aucap = average_precision_score(y_true=y, y_score=y_pred[i], pos_label=1)
    f1Score = f1_score(y_true=y, y_pred=score)
    print(f'AUC ROC for {methods[i]}: {aucroc}')
    print(f'AP PR for {methods[i]}: {aucap}')
    print(f'F1 Score for {methods[i]}: {f1Score}')
    save_result(methods[i], aucroc, aucap, f1Score, args)

if args.umap:
    import umap
    perplexity = 30
    metric='euclidean'
    min_dist = 0.1

    umap_ = umap.UMAP(random_state=RANDOM_SEED, n_neighbors=perplexity, metric=metric, min_dist=min_dist,n_components=2)
    umap_.fit(X)
    X_embedded = umap_.transform(X)

    red_true = y == 1
    green_true = y == 0
    fig, ax = plt.subplots(1,len(methods)+1)
    ax[0].scatter(X_embedded[green_true, 0], X_embedded[green_true, 1], c="g", marker='x')
    ax[0].scatter(X_embedded[red_true, 0], X_embedded[red_true, 1], c="r", marker='x')
    ax[0].set_title('Ground truth')
    
    for i, score in enumerate (y_list) :
    
        green = score == 0
        red = score == 1
    
        ax[i+1].scatter(X_embedded[green, 0], X_embedded[green, 1], c="g", marker='x')
        ax[i+1].scatter(X_embedded[red, 0], X_embedded[red, 1], c="r", marker='x')
        ax[i+1].set_title(methods[i])

    plt.savefig(os.path.join(save_path,'umap_score_baselines.png'))