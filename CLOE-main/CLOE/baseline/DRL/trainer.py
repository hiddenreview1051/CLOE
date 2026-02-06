# from https://github.com/HangtingYe/DRL/blob/main/DRL_Model/Trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from DataSet.DataLoader import get_dataloader
from model import DRL
from utils import get_logger, F1Performance, NpzDataset
from sklearn.metrics import roc_auc_score, average_precision_score
# import ipdb
import numpy as np
import matplotlib.pyplot as plt
import os

class Trainer(object):
    def __init__(self, model_config: dict):
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        self.model = DRL(model_config).to(self.device)

        dataset_name = model_config['dataset_name']
        train_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], model_config['preprocess'], mode='train',random_seed=model_config['random_seed'])
        test_set = NpzDataset(dataset_name, model_config['data_dim'], model_config['data_dir'], model_config['preprocess'], mode='eval', random_seed=model_config['random_seed'])
        self.train_loader = DataLoader(train_set,
                              batch_size=model_config['batch_size'],
                              num_workers=model_config['num_workers'],
                              shuffle=False,
                              )
        self.test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=False)
        self.model_config = model_config

    def training(self, epochs):
        train_logger = get_logger(f"CLOE/baseline/DRL/models/{self.model_config['dataset_name']}_DRL.log")
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        for epoch in range(epochs):
            running_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                # decomposition loss
                loss = self.model(x_input).mean()

                # alignment loss
                if self.model_config['input_info'] == True:
                    h = self.model.encoder(x_input)
                    x_tilde = self.model.decoder(h)
                    # s_loss = (1-F.cosine_similarity(x_tilde, x_input, dim=-1)).mean() 
                    s_loss = F.cosine_similarity(x_tilde, x_input, dim=-1).mean() * (-1)
                    loss += self.model_config['input_info_ratio'] * s_loss

                # separation loss
                if self.model_config['cl'] == True:
                    h_ = F.softmax(self.model.phi(x_input), dim=1)
                    selected_rows = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
                    h_ = h_[selected_rows]

                    matrix = h_ @ h_.T
                    mol = torch.sqrt(torch.sum(h_**2, dim=-1, keepdim=True)) @ torch.sqrt(torch.sum(h_.T**2, dim=0, keepdim=True))
                    matrix = matrix / mol
                    # d_loss =  ((1 - torch.eye(h_.shape[0]).cuda()) * matrix).sum() /(h_.shape[0]) / (h_.shape[0])
                    d_loss =  ((1 - torch.eye(h_.shape[0])) * matrix).sum() /(h_.shape[0]) / (h_.shape[0])
                    loss += self.model_config['cl_ratio'] * d_loss
                
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            train_logger.info(info.format(epoch,running_loss))
        torch.save(self.model, f"CLOE/baseline/DRL/models/{self.model_config['dataset_name']}_DRL.pth")
        print("Training complete.")
        train_logger.handlers.clear()

    def evaluate(self, umap_enable= False):
        model = torch.load(f"CLOE/baseline/DRL/models/{self.model_config['dataset_name']}_DRL.pth", weights_only=False)
        model.eval()
        mse_score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)

            h = model.encoder(x_input)

            weight = F.softmax(self.model.phi(x_input), dim=1)
            h_ = weight@model.basis_vector

            mse = F.mse_loss(h, h_, reduction='none')
            mse_batch = mse.mean(dim=-1, keepdim=True)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        # mse_rauc, mse_ap = aucPerformance(mse_score, test_label)
        mse_rauc = roc_auc_score(y_true=test_label, y_score=mse_score)
        mse_ap = average_precision_score(y_true=test_label, y_score=mse_score, pos_label=1)
        mse_f1 = F1Performance(mse_score, test_label)

        if umap_enable:
            import umap
            perplexity = 30
            metric='euclidean'
            min_dist = 0.1

            umap_ = umap.UMAP(random_state=49, n_neighbors=perplexity, metric=metric, min_dist=min_dist,n_components=2)
            umap_.fit(np.concat([x.numpy() for x, y in self.test_loader]))
            X_embedded = umap_.transform(np.concat([x.numpy() for x, y in self.test_loader]))

            y = np.concat([y.numpy() for x, y in self.test_loader])
            green_true = y == 0
            red_true = y == 1
            fig, ax = plt.subplots(1,2)
            ax[0].scatter(X_embedded[green_true, 0], X_embedded[green_true, 1], c="g", marker='x')
            ax[0].scatter(X_embedded[red_true, 0], X_embedded[red_true, 1], c="r", marker='x')
            ax[0].set_title('Ground truth')
            
            green = test_label == 0
            red = test_label == 1
            
            ax[1].scatter(X_embedded[green, 0], X_embedded[green, 1], c="g", marker='x')
            ax[1].scatter(X_embedded[red, 0], X_embedded[red, 1], c="r", marker='x')
            
            ax[1].set_title("DRL")
            fig.set_figwidth(15)
            fig.set_figheight(7)
            plt.savefig(os.path.join("CLOE/baseline/DRL/models/",'umap_score.png'))
        return mse_rauc, mse_ap, mse_f1