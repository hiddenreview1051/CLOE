import torch
import torch.optim as optim
from model import MCM
from loss import LossFunction
from score import ScoreFunction
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class Trainer(object):
    def __init__(self, run: int, model_config: dict):
        self.run = run
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        data_name = model_config['dataset_name']
        data = np.load(os.path.join(model_config['data_dir'], data_name+'.npz'), allow_pickle=True)
        X, y = data['X'], data['y'].astype(int).reshape(-1)
        x = torch.from_numpy(X)
        if data_name == "9_census":
            test_size = 1- 3000/x[y==0].shape[0]
        elif data_name == "24_mnist":
            test_size = 1- 5000/x[y==0].shape[0]
        elif x[y==0].shape[0]<8000:
            test_size = 0.1
        else:
            test_size = 1- 5000/x[y==0].shape[0]
        X_train_valid, _= train_test_split(x[y==0], test_size=test_size, random_state=model_config['random_seed'])
        X_train, _ = train_test_split(X_train_valid, test_size=0.2, random_state=model_config['random_seed'])
        self.train_loader = DataLoader(X_train, batch_size = model_config['batch_size'], num_workers = model_config['num_workers'])
        self.test_set = DataLoader(x, batch_size = 1)
        self.label = y
        model_config['data_dim'] = x.shape[1]
        self.model = MCM(model_config).to(self.device)
        self.loss_fuc = LossFunction(model_config).to(self.device)
        self.score_func = ScoreFunction(model_config).to(self.device)

    def training(self, epochs):
        # train_logger = get_logger('train_log.log')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        min_loss = 100
        for epoch in range(epochs):
            for step, x_input in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            print(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            # train_logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            if loss < min_loss:
                torch.save(self.model, 'CLOE/baseline/MCM/model.pth')
                min_loss = loss
        print("Training complete.")
        # train_logger.handlers.clear()

    def evaluate(self, mse_rauc, mse_ap, mse_f1):
        model = torch.load('CLOE/baseline/MCM/model.pth', weights_only=False)
        model.eval()
        mse_score = []
        for step, x_input in enumerate(self.test_set):
            y_label = self.label[step]
            x_pred, z, masks = self.model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch.item())
        test_label = self.label
        mse_rauc[self.run], mse_ap[self.run] = roc_auc_score(test_label, mse_score), average_precision_score(test_label, mse_score)
        normal_ratio = (test_label == 0).sum() / len(test_label)
        score = np.squeeze(mse_score)
        threshold = np.percentile(mse_score, 100 * normal_ratio)
        pred = np.zeros(len(score))
        pred[score > threshold] = 1
        mse_f1[self.run] = f1_score(test_label, pred)