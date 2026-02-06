# from https://github.com/JXYin24/MCM/blob/main/main.py
import argparse
import torch
import numpy as np
import os
from trainer import Trainer
model_config = {
    'dataset_name': '15_Hepatitis',
    'epochs': 200,
    'learning_rate': 0.05,
    'sche_gamma': 0.98,
    'mask_num': 15,
    'lambda': 5,
    'device': 'cpu',
    'data_dir': 'CLOE/datasets',
    'runs': 1,
    'batch_size': 512, 
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 256,
    'z_dim': 128,
    'mask_nlayers': 3,
    'random_seed': 49,
    'num_workers': 0
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=49,
                help='seed')
    parser.add_argument('--data-name', type=str, default='15_Hepatitis',
                help='Name of the dataset, must be a numpy file in configs[\'data_dir\']')
    args = parser.parse_args()
    model_config['random_seed'] = args.seed
    model_config['dataset_name'] = args.data_name
    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')
    result = []
    runs = model_config['runs']
    mse_rauc, mse_ap, mse_f1 = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(runs):
        trainer = Trainer(run=i, model_config=model_config)
        trainer.training(model_config['epochs'])
        trainer.evaluate(mse_rauc, mse_ap, mse_f1)
    mean_mse_auc , mean_mse_pr , mean_mse_f1 = np.mean(mse_rauc), np.mean(mse_ap), np.mean(mse_f1)

    print('##########################################################################')
    print("mse: average AUC-ROC: %.4f  average AUC-PR: %.4f"
          % (mean_mse_auc, mean_mse_pr))
    print("mse: average f1: %.4f" % (mean_mse_f1))
    result_path = f"CLOE/results/{args.data_name}/{args.seed}/MCM/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "AUC ROC": mse_rauc,
                "AP AUC": mse_ap,
                "F1 Score": mse_f1,
            },
        )
