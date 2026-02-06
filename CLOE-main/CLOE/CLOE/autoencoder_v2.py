from torch import nn
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle as pkl
from pathlib import Path
import os
import psutil

class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    dropout_rate [float] : dropout rate 
    """

    def __init__(self, in_shape, enc_shape, DIM, dropout_rate = 0.2):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, DIM[0]),
            nn.ReLU(True),
            nn.Dropout(dropout_rate, inplace=False))
        for i in range(len(DIM)-1):
            self.encode.add_module(f'conv_{i}', nn.Linear(DIM[i], DIM[i+1]))
            self.encode.add_module(f'relu_{i}', nn.ReLU(True))
            self.encode.add_module(f'dropout_{i}', nn.Dropout(dropout_rate, inplace=False))
        self.encode.add_module(f'conv_{i+1}', nn.Linear(DIM[-1], enc_shape))
        self.encode.add_module(f'relu_{i+1}', nn.ReLU(True))
        self.encode.add_module(f'norm_{i+1}',nn.BatchNorm1d(enc_shape))
        self.encode.add_module(f'tanh_{i+1}', nn.Tanh())
        
        self.decode = nn.Sequential(
            nn.Linear(enc_shape, DIM[-1]),
            nn.ReLU(True),
            nn.Dropout(dropout_rate, inplace=False))
        for j in range(len(DIM)-1):
            self.decode.add_module(f'deconv_{j}', nn.Linear(DIM[len(DIM)-1-j], DIM[len(DIM)-1-(j+1)]))
            self.decode.add_module(f'relu_{j}', nn.ReLU(True))
            self.decode.add_module(f'dropout_{j}', nn.Dropout(dropout_rate, inplace=False))
        self.decode.add_module(f'conv_{j+1}', nn.Linear(DIM[0], in_shape))
        self.decode.add_module(f'sigmoid', nn.Sigmoid())

    def forward(self, x):
        x_enc = self.encode(x)
        x = self.decode(x_enc)
        return x, x_enc

def compute_grad_norm(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=False, allow_unused=True)
    total_norm = 0
    for g in grads:
        if g is not None:
            total_norm += g.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train(model, train_loader, validation_loader, epochs, device, optimizer, loss_function, train_mode, path_save, patience_max=50,  verbose = True, abl3= False):
    """
    Train the AE model.

    Args:
        model (nn.Module): The AE model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training batches.
        validation_loader (torch.utils.data.DataLoader): DataLoader providing the validation batches.
        epochs (int): The number of epochs to train the model.
        device (torch.device): The device (CPU or GPU) on which to perform the computation.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        loss_function (nn.Module): The loss function to use for training.
        train_mode (int): 0 for pre-training, 1 for joint training.
        path_save (str): path to save the trained model.
        patience_max (int):patience for the early stopping training procedure.
    """
    dict_info={}
    train_losses_cf = []
    train_losses_cf_mse = []
    train_losses_cf_cloe = []
    validation_losses_cf = []
    validation_losses_cf_mse = []
    validation_losses_cf_cloe = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_loss_v = float('inf')
    patience = 0
    memory_list = []

    for epoch in range(epochs):
        model.to(device)
        # Training phase
        overall_loss = 0
        overall_loss_mse = 0
        overall_loss_cloe = 0
        model.train()
        
        for x in train_loader:
            x = x.to(device)
            output, x_enc = model(x)
            assert output.requires_grad
            # dataset = model.encode(torch.concatenate([x for x in train_loader]))
            loss, loss_mse, loss_cloe = loss_function(output, x, x_enc, x_enc, valid = False)
            overall_loss_mse += loss_mse.item()
            overall_loss_cloe += loss_cloe.item()
            if loss_cloe.item() > 0:
                grad_norm_MSE = compute_grad_norm(loss_mse, model)
                grad_norm_CLOE = compute_grad_norm(loss_cloe, model)
                # print(f"Gradient norm of CLOE loss: {grad_norm_CLOE:.4f}")
                if  grad_norm_CLOE != 0 and abl3:
                    lambda_CLOE = (grad_norm_MSE / grad_norm_CLOE)
                elif grad_norm_CLOE > 0:
                    lambda_CLOE = (grad_norm_MSE / grad_norm_CLOE) *100
                else:
                    lambda_CLOE = 1.0  # fallback to avoid div by zero
                loss = loss_mse + lambda_CLOE * loss_cloe
                loss.backward()
            else:
                loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(f"WARNING: {name} has zero grad!")
            optimizer.step()
            optimizer.zero_grad()
            overall_loss += loss.item()

        process = psutil.Process(os.getpid())

        mem_bytes = process.memory_info().rss  # resident set size, in bytes
        mem_MB = mem_bytes / (1024 ** 2)       # convert to MB
        memory_list.append(mem_MB)
        # print(f"CPU memory used: {mem_MB:.2f} MB")

        # Validation phase
        model.eval()
        with torch.no_grad():
            dataset =  model.encode(torch.concatenate([x for x in train_loader])) # compute the support of Christoffel function with all the training data
            output = torch.concatenate([model(y)[0] for y in validation_loader])
            y_enc = torch.concatenate([model(y)[1] for y in validation_loader])
            v_loss, v_loss_mse, v_loss_cloe = loss_function(output = output, x = torch.concatenate([y for y in validation_loader]), x_enc = y_enc, dataset = dataset, valid = True)
            validation_losses_cf.append(v_loss.item())
        
        avg_train_loss = overall_loss / len(train_loader)
        train_losses_cf.append(avg_train_loss)
        avg_validation_loss = v_loss
        train_losses_cf_mse.append(overall_loss_mse/ len(train_loader))
        train_losses_cf_cloe.append(overall_loss_cloe/ len(train_loader))
        validation_losses_cf_mse.append(v_loss_mse)
        validation_losses_cf_cloe.append(v_loss_cloe)

        # Save the best model
        if avg_validation_loss < best_loss_v and epoch>5:
            best_loss_v = avg_validation_loss
            epoch_best_loss_v = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            dict_info['epoch_v']=epoch
            dict_info['best_loss_v']=best_loss
            patience = 0
        if avg_train_loss < best_loss and epoch>5:
            best_loss = avg_train_loss
            epoch_best_loss = epoch
            dict_info['epoch']=epoch
            dict_info['best_loss']=best_loss
            patience = 0
        else :
            patience += 1
        
        if patience > patience_max:
            break 

        if verbose and epoch % int(0.1*epochs) == 0:
            print(f'epoch {epoch} \t validation loss: {avg_validation_loss}')
            print(f'epoch {epoch} \t training loss: {avg_train_loss}')
    dict_info['loss training christoffel'] = train_losses_cf
    dict_info['loss validation christoffel'] = validation_losses_cf
    print(f'final epoch is {epoch_best_loss_v} with a validation loss of {best_loss_v}')
    print(f'final epoch is {epoch_best_loss} with a training loss of {best_loss}')

    # Load and save the best weights of the best model
    model.load_state_dict(best_model_wts)
    if train_mode == 1 :
        file_save =  os.path.join(path_save,'jointrained')
    else :
        file_save = os.path.join(path_save,'pretrain')

    Path(path_save).mkdir(parents=True, exist_ok=True)   
    torch.save(model.state_dict(),f'{file_save}.pt')
    with open(f'{file_save}.pickle', 'wb') as handle:
        pkl.dump(dict_info, handle, protocol=pkl.HIGHEST_PROTOCOL)

    fig, ax = plt.subplots(1,2)
    ax[0].scatter(range(epoch+1), train_losses_cf, label='Train Loss Christoffel', color='blue')
    ax[1].scatter(range(epoch+1), validation_losses_cf, label='Validation Loss Christoffel', color='orange')

    fig.set_figwidth(10)
    fig.set_figheight(6)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Christoffel Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Christoffel Loss')
    fig.set_label('Train and Validation Christoffel Loss')
    ax[0].plot()
    ax[1].plot()
    plt.legend()
    plt.savefig(f"{file_save}.png")

    fig, ax = plt.subplots(1,2)
    ax[0].scatter(range(epoch+1), train_losses_cf_mse, label='Train MSE loss', color='blue')
    ax[1].scatter(range(epoch+1), train_losses_cf_cloe, label='Train CLOE loss', color='blue')

    fig.set_figwidth(10)
    fig.set_figheight(6)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Christoffel Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Christoffel Loss')
    fig.set_label('Train Christoffel Losses')
    ax[0].plot()
    ax[1].plot()
    plt.legend()
    plt.savefig(f"{file_save}_mse&cloe.png")

    fig, ax = plt.subplots(1,2)
    ax[0].scatter(range(epoch+1),validation_losses_cf_mse, label='Valid MSE loss', color='orange')
    ax[1].scatter(range(epoch+1), validation_losses_cf_cloe, label='Valid CLOE loss', color='orange')

    fig.set_figwidth(10)
    fig.set_figheight(6)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Christoffel Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Christoffel Loss')
    fig.set_label('Validation Christoffel Losses')
    ax[0].plot()
    ax[1].plot()
    plt.legend()
    plt.savefig(f"{file_save}_mse&cloe_valid.png")
    return memory_list


