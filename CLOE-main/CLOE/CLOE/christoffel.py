from base import BaseDetector
import torch
from utils import MomentsMatrix
from math import comb
from torch import nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class CLOE(BaseDetector):
    """
    Guided autoencoder with the Christoffel function

    Attributes
    ----------
    n: int
        the degree of polynomials, usually set between 2 and 6
    regularization: str, optional
        one of  "max" (score divided by the maximum value of the Christoffel function on the training set), "vu" (score divided by d^{3p/2}) as DyCF, "comb" (score divided by comb(p+d, d)) or "none" (no regularization), (default is "max")
    polynomial_basis: str, optional
        polynomial basis used to compute moment matrix, either "monomials", "chebyshev_t_1", "chebyshev_t_2", "chebyshev_u" or "legendre",
        varying this parameter can bring stability to the score in some cases (default is "monomials")
    inv: str, optional
        inversion method, one of "inv" for classical matrix inversion, "pinv" for Moore-Penrose pseudo-inversion or "fpd_inv" for Cholesky inversion (default is "fpd_inv")

    Methods
    -------
    See BaseDetector methods
    """

    def __init__(self, n: int = None, regularization: str = "max", polynomial_basis: str = "monomials", inv: str = "fpd_inv", device='cpu'):
        self.n = n # Dimension of the polynomial
        self.device = device
        self.regularization = regularization
        self.regularizer = None
        self.polynomial_basis = polynomial_basis
        self.inv = inv
        

    def fit(self, x: torch.Tensor, x_valid:torch.Tensor=None):
        self.assert_shape_unfitted(x)
        self.N = x.shape[0]  # Number of samples
        self.d = x.shape[1]  # Dimension of the data
        self.n = self.n if self.n != None else self.compute_nN(x) # dimension of the polynome
        self.moments_matrix = MomentsMatrix(self.n, polynomial_basis=self.polynomial_basis, inv_opt=self.inv, device = self.device)
        self.moments_matrix.fit(x)
        if self.regularization == "vu":
            self.regularizer = self.n ** (3 * self.d / 2)
        elif self.regularization == "max":
            score_samples = self.score_samples_noreg(x)
            self.regularizer = torch.max(score_samples)
        elif self.regularization == "comb":
            self.regularizer = comb(self.n + x.shape[1], x.shape[1])
        elif self.regularization == "cross_valid":
            score_samples = self.score_samples_noreg(x)
            assert x_valid != None
            if x_valid.shape[0]>200:
                score_samples_valid = self.score_samples_noreg(x_valid)
                self.regularizer = self.compute_threshold(score_samples, score_samples_valid)
            else : 
                self.regularizer = torch.max(score_samples)
        else:
            self.regularizer = self.regularization
        return self

    def update(self, x):
        self.assert_shape_fitted(x)
        self.moments_matrix.update(x, self.N)
        self.N += x.shape[0]
        return self
    
    def score_samples(self, x):
        self.assert_shape_fitted(x)
        return self.moments_matrix.score_samples(x.reshape(-1, self.d)) / self.regularizer
    
    def score_samples_noreg(self, x):
        self.assert_shape_fitted(x)
        return self.moments_matrix.score_samples(x.reshape(-1, self.d))

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return (1 / self.score_samples(x)) - 1

    def predict(self, x):
        self.assert_shape_fitted(x)
        return torch.where(self.decision_function(x) < 0, 1, 0)

    def fit_predict(self, x):
        self.assert_shape_fitted(x)
        self.fit(x.reshape(-1, self.d))
        return self.predict(x.reshape(-1, self.d))

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = torch.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            evals[i] = self.decision_function(xx.reshape(-1, self.d))
            self.update(xx.reshape(-1, self.d))
        return evals

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        preds = torch.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            preds[i] = self.predict(xx.reshape(-1, self.d))
            self.update(xx.reshape(-1, self.d))
        return preds

    def save_model(self):
        return {
            "N": self.N,
            "d": self.d,
            "n": self.n,
            "regularizer": self.regularizer,
            "regularization": self.regularization,
            "moments_matrix": self.moments_matrix.save_model()
        }

    def load_model(self, model_dict: dict):
        self.N = model_dict["N"]
        self.d = model_dict["d"]
        self.n = model_dict["n"]
        self.regularizer = model_dict["regularizer"]
        self.regularization = model_dict["regularization"]
        self.moments_matrix = self.moments_matrix.load_model(model_dict["moments_matrix"])


    def copy(self):
        c_bis = CLOE(n=self.n, regularization=self.regularization)
        c_bis.moments_matrix = self.moments_matrix.copy()
        c_bis.N = self.N
        if self.d is not None:
            c_bis.d = self.d
        if self.regularizer is not None:
            c_bis.regularizer = self.regularizer
        return c_bis

    def method_name(self):
        return "CLOE"
    
    # Vu et al. (2019) 4.1
    def compute_nN(self, x):
        list_d = []
        for n_test in range(2,6):
            s_d = comb(n_test+self.d, n_test)
            if s_d < self.N :
                christoffel_test = CLOE(n_test, "max", self.polynomial_basis).fit(x)
                score = 1/christoffel_test.score_samples_noreg(x)
                list_d.append(1/self.N*s_d*sum(score))
        n = list_d.index(max(list_d))+2
        print(f"Computed n according to section 4.1 from Vu et al. is: {n+1}")
        return(max(n+1,6))
    
    def compute_threshold(self, scores_x, scores_valid):
        quantiles = torch.tensor([0.1*i for i in range(1, 10)], dtype=scores_x.dtype)
        thresholds =  list(torch.quantile(scores_x, quantiles)) + [torch.max(scores_x)]
        y_true = [0 for i in range(scores_valid.shape[0])]
        perf = []
        for thresh in thresholds:
            y_pred = [1 if score >= thresh else 0 for score in scores_valid]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            accuracy = (tp + tn) / len(y_true)
            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)
            print(f"Threshold {thresh:.2f}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
            print(f"FPR: {fpr:.3f}, FNR: {fnr:.3f}\n")
            perf.append(fpr)
        threshold = thresholds[perf.index(max(perf))]
        print(threshold)
        print(perf.index(min(perf)))
        return(threshold)
    

class ChristoffelScore_loss(nn.Module):
    def __init__(self, n: float, step_training: int, type_conc:str ='sum', regularization:str = 'comb', random_seed:int = 49, device:str ='cpu', lambda_CLOE:int = 1):
        super().__init__()
        self.step_training = step_training
        self.n = n
        self.type_conc = type_conc
        self.regularization = regularization
        self.random_seed = random_seed
        self.MSE = nn.MSELoss()
        self.device = device
        self.lambda_CLOE = lambda_CLOE

    
    def forward(self, output, x, x_enc, dataset, valid = False):
        mse_loss = self.MSE(output, x)
        if self.step_training == 0 : 
            return mse_loss, mse_loss, torch.zeros(1)
        else : 
            if not x_enc.requires_grad:
                    x_enc = x_enc.clone().detach().requires_grad_(True)
            if valid :
                X_support = dataset # compute the support with all training data
            else : 
                batch_size = x_enc.shape[0]
                indices = torch.randperm(batch_size, device=x_enc.device)
                split_idx = int(batch_size * 0.8)
                support_idx = indices[:split_idx]
                X_support = x_enc[support_idx]    
            score = CLOE(self.n, polynomial_basis = "monomials", inv = 'fpd_inv', device = self.device).fit(X_support).score_samples_noreg(x_enc)
            # score = score / torch.max(score)
            if self.type_conc == 'sum' :
                return mse_loss + self.lambda_CLOE * torch.sum(score)
            elif self.type_conc == 'max':
                return mse_loss + self.lambda_CLOE * torch.max(score)
            elif self.type_conc == 'min':
                return mse_loss + self.lambda_CLOE * torch.min(score)
            else : # If not sum or max or min, it is mean
                return  mse_loss + self.lambda_CLOE * torch.mean(score),  mse_loss ,  torch.mean(score)
