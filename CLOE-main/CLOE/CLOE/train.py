import torch
import numpy as np
import math as m
import pickle as pkl
import argparse
import time
import os

from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from christoffel import CLOE, ChristoffelScore_loss
from autoencoder_v2 import train, Autoencoder

def print_v(args, string):
    """
        Print a message if verbosity is enabled in args.
    """
    if args.verbose :
        print(string)


def get_parser():
    """
    Returns an argument parser for training options.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=49,
                        help='seed')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--n', type=int, default=2,
                        help='degree of the poynomial to compute the support')
    parser.add_argument('--nb-epochs', type=int, default=150,
                        help='Number of epochs for this training step')
    parser.add_argument('--nb-class', type=int, default=8,
                        help='Dimension of the latent space of the autoencoder')
    parser.add_argument('--dataset-path', type=str, default='CLOE/datasets/',
                        help='Path to the dataset folder (numpy file)')
    parser.add_argument('--save-path', type=str, default='CLOE/datasets/models/',
                        help='Path to save the model')
    parser.add_argument('--data-name', type=str, default='15_Hepatitis',
                        help='Name of the dataset (numpy file)')
    parser.add_argument('--training-step', type=str, default='pre-training',
                        help='\'pre-training\' or \'joint-training\' or \'compute-support\' accepted')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for earlystopping')
    parser.add_argument('--lambda_CF', type=int, default=1,
                        help='Coefficient lambda in the training loss for the Christoffel function part')
    parser.add_argument('--dim', type=int, default=[500, 500, 2000], nargs='+',
                        help='Dimension of the hidden layer of the encoder in the order')
    parser.add_argument('--type-conc', type=str, default='mean',
                        help='Type of the concatenation for all the Christoffel value in the loss : mean, sum or max')
    parser.add_argument('--num-worker', type=int, default=0,
                        help='Number of worker used to train the model')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Print the log in console')
    
    return(parser)

def main():
    args = get_parser().parse_args()

    # Set the random seed for reproducibility

    torch.manual_seed(args.seed)

    # Choose CPU or GPU if available automatically
    if torch.cuda.is_available():
        print_v(args, 'Using Cuda')
        torch.cuda.manual_seed(args.seed)
        args.device = 'cuda:0'
    else : 
        args.device = 'cpu'

    time_begin = time.time()


    # Hyperparameters
    RANDOM_SEED = args.seed
    LEARNING_RATE = args.lr
    n = args.n
    NUM_EPOCHS = args.nb_epochs
    NUM_CLASSES = args.nb_class
    
    training_step = args.training_step
    type_conc = args.type_conc
    lambda_CLOE = args.lambda_CF
    DIM = args.dim

    save_path = os.path.join(args.save_path, args.data_name)

    # Enable multiprocessing
    NUM_WORKER = args.num_worker
    if NUM_WORKER > 1 :
        torch.set_num_threads(NUM_WORKER) 
        torch.set_num_interop_threads(NUM_WORKER) 

    # Dataset preprocessing
    data_name = args.data_name
    data = np.load(f'{args.dataset_path}{data_name}.npz', allow_pickle=True)
    X, y = data['X'], data['y']
    if data_name == "24_mnist":
        x = torch.from_numpy(RobustScaler().fit_transform(X)).to(args.device)
    else:
        x = torch.from_numpy(StandardScaler().fit_transform(X)).to(args.device)
    print_v(args, f'Number of instances : {X.shape[0]}, number of dimension: {X.shape[1]}')

    # Compute automatically the batch size depending of n and N
    BATCH_SIZE = m.comb(NUM_CLASSES+n, n)
    print_v(args, f'Batch size is: {BATCH_SIZE}')

    train_mode = {
        "pre-training": 0,
        "joint-training": 1,
        "compute-support": 2
    }
    assert training_step in train_mode.keys()
    if train_mode[training_step] == 0:
        dropout_rate = 0.2
    else :
        dropout_rate = 0
    print_v(args, f'The dropout rate used to train this model is: {dropout_rate}')
    autoencoder = Autoencoder(in_shape=x.shape[1], enc_shape=NUM_CLASSES, DIM = DIM, dropout_rate = dropout_rate).double().to(args.device)
    print_v(args, autoencoder)

    if train_mode[training_step] == 1 : 
        file_save = os.path.join(save_path,'pretrain')
        autoencoder.load_state_dict(torch.load( f'{file_save}.pt'))
        print_v(args, 'pre-trained model load')
    if train_mode[training_step] == 2 : 
        file_save = os.path.join(save_path,'jointrained')
        autoencoder.load_state_dict(torch.load( f'{file_save}.pt'))
        print_v(args, 'joint-trained model load')

    error = ChristoffelScore_loss(n=n, type_conc=type_conc, step_training=train_mode[training_step], random_seed = RANDOM_SEED, device = args.device, lambda_CLOE = lambda_CLOE) 

    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    if data_name == "9_census":
        test_size = 1- 3000/x[y==0].shape[0]
    elif data_name == "24_mnist":
        test_size = 1- 5000/x[y==0].shape[0]
    elif x[y==0].shape[0]<8000:
        test_size = 0.1
    else:
        test_size = 1- 8000/x[y==0].shape[0]
    X_train_valid, X_test= train_test_split(x[y==0], test_size=test_size, random_state=RANDOM_SEED)
    print_v(args,f'Test set shape: {X_test.shape}')
    X_train, X_valid = train_test_split(X_train_valid,test_size=0.2, random_state=RANDOM_SEED)
    print_v(args,f'Train set shape: {X_train.shape}')
    print_v(args,f'Valid set shape: {X_valid.shape}')

    if train_mode[training_step] == 0 or train_mode[training_step] == 1 :

        train(
            model              = autoencoder,
            train_loader       = DataLoader(X_train.to(dtype=torch.float64), batch_size = BATCH_SIZE, num_workers = NUM_WORKER),
            validation_loader  = DataLoader(X_valid.to(dtype=torch.float64)),
            epochs             = NUM_EPOCHS,
            device             = args.device,
            optimizer          = optimizer, 
            loss_function      = error, 
            patience_max       = args.patience,
            train_mode         = train_mode[training_step],
            path_save          = save_path,)

    else : 
        with torch.no_grad():
            autoencoder.eval()
            x_encoded = autoencoder.encode(X_train.to(dtype=torch.float64))
        christoffel_support = CLOE(n=n, regularization= "max", polynomial_basis = "monomials", inv = 'fpd_inv', device = args.device)
        christoffel_support.fit(x_encoded)

        with torch.no_grad():
            x_encoded = autoencoder.encode(x.to(dtype=torch.float64))

        score = christoffel_support.predict(x_encoded)

        from sklearn.metrics import roc_auc_score, average_precision_score 

        aucroc = roc_auc_score(y_true=y, y_score=score.cpu())
        aucap = average_precision_score(y_true=y, y_score=score.cpu(), pos_label=1)

        print(f'AUC ROC for Christoffel score: {aucroc}')
        print(f'AP AUC for Christoffel score: {aucap}')

        with open(os.path.join(save_path,'jointrained.pkl'), 'wb') as f:
            pkl.dump(christoffel_support.save_model(), f)
    print(f'time: {time.time()-time_begin}')
        

if __name__ == "__main__":
    main()