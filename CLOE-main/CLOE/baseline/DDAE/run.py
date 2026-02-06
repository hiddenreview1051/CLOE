# from https://github.com/sattarov/AnoDDAE/blob/v0.1.0/run.py
import argparse
import yaml
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from src.model import AnomalyDetector
from data import load_data, split_data
from utils import set_seed, normalize_data, evaluate_anomaly_detection, get_batch_size
from model import DDAE, DiffusionScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly detection experiment.")
    parser.add_argument('--config', type=str, default='CLOE/baseline/DDAE/config.yaml',
                        help='Path to the config file.')
    parser.add_argument('--seed', type=int, default=49,
                        help='Random seed.')
    parser.add_argument('--data-name', type=str, default='15_Hepatitis',
                        help='Name of the dataset, must be a numpy file in configs[\'data_dir\']')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed #config.get('seed', 49)

    # Set seed for reproducibility
    set_seed(seed)

    # Load data
    data_dir = config['data']['path']
    data = np.load(os.path.join(data_dir,args.data_name+'.npz'), allow_pickle=True)
    X, y = data['X'], data['y']
    x = torch.from_numpy(X).to(torch.float32) 
    print(args, f'Number of instances : {X.shape[0]}, number of dimension: {X.shape[1]}')

    if args.data_name == "9_census":
        test_size = 1- 3000/x[y==0].shape[0]
    elif args.data_name == "24_mnist":
        test_size = 1- 5000/x[y==0].shape[0]
    elif x[y==0].shape[0]<8000:
        test_size = 0.1
    else:
        test_size = 1- 8000/x[y==0].shape[0]
    X_train_valid, x_test= train_test_split(x[y==0], test_size=test_size, random_state=seed)
    x_train, x_valid = train_test_split(X_train_valid, test_size=0.2, random_state=seed)
    print(args,f'Train set shape: {x_train.shape}, valid set shape: {x_valid.shape} and test set shape: {x_test.shape}')

    # convert to tensors
    # x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = x
    y_train = None #torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y, dtype=torch.float32)

    # Initialize and train model
    model = DDAE(
        input_dim=x_train.shape[1],
        hidden_dim=config['model']['hidden_dim'],
        activation=config['model']['activation'],
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        scheduler=config['diffusion']['scheduler'],
        time_emb_dim=config['diffusion']['time_emb_dim'],
        time_emb_type=config['diffusion']['time_emb_type'],
        epochs=config['train']['epochs'],
        batch_size=get_batch_size(X.shape[0]),
        learning_rate=config['train']['lr'],
        eval_epochs= config['train']['eval_epochs'],
        )
    print("Batch size:", get_batch_size(X.shape[0]))
    model.fit(x_train, x_test, y_train, y_test)

    # Predict anomaly scores
    scores = model.predict(x_test)

    # Evaluate
    results = evaluate_anomaly_detection(scores=scores.numpy(), labels=y_test.numpy())
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    
    print('##########################################################################')
    print("AUC-ROC: %.4f  AUC-PR: %.4f"
          % (results["ROC-AUC"], results["AP"]))

    result_path = f"CLOE/results/{args.data_name}/{args.seed}/DDAE/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "AUC ROC": results["ROC-AUC"],
                "AP AUC": results["AP"],
            },
        )

if __name__ == '__main__':
    main()