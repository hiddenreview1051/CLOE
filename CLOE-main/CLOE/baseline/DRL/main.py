# https://github.com/HangtingYe/DRL/blob/main/main.py
import torch
import numpy as np
import argparse
import os
from scipy import io
import importlib
from sklearn.cluster import KMeans
import glob
# import ipdb
import time

npz_files = glob.glob(os.path.join('CLOE/datasets', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

# mat_files = glob.glob(os.path.join('./Data', '*.mat'))
# mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

if __name__ == "__main__":

    time_begin = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--preprocess', type=str, default='standard')
    parser.add_argument('--diversity', type=str, default='True')
    parser.add_argument('--plearn', type=str, default='False')
    parser.add_argument('--input_info', type=str, default='True')
    parser.add_argument('--input_info_ratio', type=float, default=0.1)
    parser.add_argument('--cl', type=str, default='True')
    parser.add_argument('--cl_ratio', type=float, default=0.06)
    parser.add_argument('--basis_vector_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=200)
    # parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    diversity = True if args.diversity == 'True' else False
    plearn = True if args.plearn == 'True' else False
    input_info = True if args.input_info == 'True' else False
    cl = True if args.cl == 'True' else False

    dict_to_import = 'model_config_'+args.model_type
    module_name = 'configs'
    module = importlib.import_module(module_name)
    model_config = getattr(module, dict_to_import)

    model_config['preprocess'] = args.preprocess
    model_config['diversity'] = diversity
    model_config['plearn'] = plearn
    model_config['input_info'] = input_info
    model_config['input_info_ratio'] = args.input_info_ratio
    model_config['cl'] = cl
    model_config['cl_ratio'] = args.cl_ratio
    model_config['random_seed'] = args.seed
    model_config['epochs'] = args.epoch

    torch.manual_seed(model_config['random_seed'])
    torch.cuda.manual_seed(model_config['random_seed'])
    np.random.seed(model_config['random_seed'])
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn')

    # if args.dataname in npz_datanames:
    path = os.path.join(model_config['data_dir'], args.dataname + '.npz')
    data = np.load(path)
    # elif args.dataname in mat_datanames:
    #     path = os.path.join(model_config['data_dir'], args.dataname + '.mat')
    #     data = io.loadmat(path)
    samples = data['X']
    model_config['dataset_name'] = args.dataname
    model_config['data_dim'] = samples.shape[-1]

    if args.model_type == 'DRL':
        from trainer import Trainer
        model_config['basis_vector_num'] = args.basis_vector_num

    trainer = Trainer(model_config=model_config)
    trainer.training(model_config['epochs'])
    print(f'time training: {time.time()-time_begin}')
    mse_rauc, mse_ap, mse_f1 = trainer.evaluate()

    print('##########################################################################')
    print("AUC-ROC: %.4f  AUC-PR: %.4f"
          % (mse_rauc, mse_ap))
    print("f1: %.4f" % (mse_f1))

    result_path = f"CLOE/results/{args.dataname}/{args.seed}/DRL/"
    os.makedirs(result_path, exist_ok=True)
    np.save(
            result_path + "result.npy",
            {
                "AUC ROC": mse_rauc,
                "AP AUC": mse_ap,
                "F1 Score": mse_f1,
            },
        )

    