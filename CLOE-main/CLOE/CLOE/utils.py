# from https://github.com/kyducharlet/odds/blob/main/odds/utils.py, all numpy operations have been remplaced by torch operations to enable differentiability through the CF

import torch
import math as m
from math import comb, factorial

inds_cache = {}

""" Polynomials basis functions """

def monomials(x, n):
    x_repeated = torch.tile(x, (n.shape[0], 1))
    return torch.pow(x_repeated, n)


def chebyshev_t_1(x, n_matrix):
    x_repeated = torch.tile(x, (n_matrix.shape[0], 1))

    # Mask for each conditions on x
    mask_1 = x_repeated < -1
    mask_2 = x_repeated > 1
    mask_3 = (x_repeated >= -1) & (x_repeated <= 1)

    # initialize result
    result = torch.zeros_like(n_matrix, dtype=float)

    # Compute for each conditions
    # x < -1
    result[mask_1] = ((-1) ** n_matrix[mask_1] *
                      torch.cosh(n_matrix[mask_1] * torch.arccosh(-x_repeated[mask_1])) /
                      torch.where(n_matrix[mask_1] == 0, torch.sqrt(torch.tensor(m.pi)), torch.sqrt(torch.tensor(m.pi) / 2)))

    # x > 1
    result[mask_2] = (torch.cosh(n_matrix[mask_2] * torch.arccosh(x_repeated[mask_2])) /
                      torch.where(n_matrix[mask_2] == 0, torch.sqrt(torch.tensor(m.pi)), torch.sqrt(torch.tensor(m.pi) / 2)))

    # -1 <= x <= 1
    result[mask_3] = (torch.cos(n_matrix[mask_3] * torch.arccos(x_repeated[mask_3])) /
                      torch.where(n_matrix[mask_3] == 0, torch.sqrt(torch.tensor(m.pi)), torch.sqrt(torch.tensor(m.pi) / 2)))

    return result


def chebyshev_t_2(x, n):  # Orthonormalized on [-1, 1] according to the Lebesgue measure with 1 / sqrt(1 - x**2) as weight
    if n == 0:
        return 1 / torch.sqrt(torch.tensor(m.pi))
    else:
        return (n / torch.sqrt(torch.tensor(m.pi) / 2)) * torch.sum([(-2)**i * (factorial(n + i - 1) / (factorial(n - i) * factorial(2 * i))) * (1 - x)**i for i in range(n+1)])


def chebyshev_u(x, n):
    if n == 0:
        return torch.sqrt(2 / torch.tensor(m.pi))
    else:
        return torch.sqrt(2 / torch.tensor(m.pi)) * torch.sum([(-2)**i * (factorial(n + i - 1) / (factorial(n - i) * factorial(2 * i + 1))) * (1 - x)**i for i in range(n+1)])


def legendre(x, n):  # Orthonormalized on [-1, 1] according to the Lebesgue measure
    return torch.sqrt((2*n + 1) / 2) * torch.sum([comb(n, i) * comb(n+i, i) * ((x - 1) / 2)**i for i in range(n+1)])


IMPLEMENTED_POLYNOMIAL_BASIS = {
    "monomials": monomials,
    "legendre": legendre,
    "chebyshev": chebyshev_u,
    "chebyshev_t_1": chebyshev_t_1,
    "chebyshev_t_2": chebyshev_t_2,
    "chebyshev_u": chebyshev_u,
}


""" Incrementation options """

def inverse_increment(mm, x, n, inv_opt):
    moments_matrix = n * mm.moments_matrix
    for xx in x:
        v = mm.polynomial_func(xx, mm.monomials)
        moments_matrix = moments_matrix + v @ v.T
    moments_matrix /= (n + x.shape[0])
    mm.moments_matrix = moments_matrix
    mm.inverse_moments_matrix = inv_opt(moments_matrix)

""" Matrix inversion options """

def pd_inv(M):
    """Positive definite matrix inversion with PyTorch while supporting gradients."""
    try:
        inv = torch.linalg.inv(M)
    except RuntimeError:
        # Regularize a bit and try again
        eye = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
        inv = torch.linalg.inv(M + 1e-10 * eye)
    return inv

def fpd_inv(M):
    """Fast positive definite inversion via Cholesky decomposition."""
    n = M.size(0)
    eye = torch.eye(n, device=M.device, dtype=M.dtype)
    try:
        cholesky = torch.linalg.cholesky(M)
        inv = torch.cholesky_inverse(cholesky)
        inv = (inv + inv.T) / 2
    except RuntimeError:
        try:
            cholesky = torch.linalg.cholesky(M + eye * 1e-10)
            inv = torch.cholesky_inverse(cholesky)
            inv = (inv + inv.T) / 2
        except RuntimeError:
            print("Error in Cholesky, falling back to pinv.")
            inv = torch.linalg.pinv(M)
    return inv

def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = torch.triu(torch.zeros((n,n),dtype=torch.bool))
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]


IMPLEMENTED_INVERSION_OPTIONS = {
    "inv": torch.linalg.inv,
    "pinv": torch.linalg.pinv,
    "pd_inv": pd_inv,
    "fpd_inv": fpd_inv,
}

""" Moments matrix """

class MomentsMatrix:
    def __init__(self, n, polynomial_basis="monomials", inv_opt="inv", device = 'cpu'):
        assert polynomial_basis in IMPLEMENTED_POLYNOMIAL_BASIS.keys()
        assert inv_opt in IMPLEMENTED_INVERSION_OPTIONS.keys()
        self.n = n
        self.device = device
        self.polynomial_func = lambda x, m: PolynomialsBasis.apply_combinations(x, m, IMPLEMENTED_POLYNOMIAL_BASIS[polynomial_basis], device)
        self.incr_func = inverse_increment
        self.inv_func = IMPLEMENTED_INVERSION_OPTIONS[inv_opt]
        self.monomials = None
        self.moments_matrix = None
        self.inverse_moments_matrix = None

    def fit(self, x):
        monomials = PolynomialsBasis.generate_combinations(self.n, x.shape[1])
        moments_matrix = torch.zeros((len(monomials), len(monomials)), dtype=x.dtype, device = x.device)
        self.monomials = torch.asarray(monomials, dtype=x.dtype, device = x.device)
        v_matrix_list = []
        for xx in x:
            v = self.polynomial_func(xx, self.monomials)
            v_matrix_list.append(v)
        v_matrix = torch.cat(v_matrix_list, dim=1)  # shape (num_features, num_samples)
        # Compute moments matrix as sum of outer products (vectorized)
        moments_matrix = v_matrix @ v_matrix.T  # shape (num_features, num_features)
        self.moments_matrix = (moments_matrix / x.shape[0]).requires_grad_(True)
        self.inverse_moments_matrix = self.inv_func(self.moments_matrix)
        return self

    def score_samples(self, x):
        res = []
        for xx in x:
            v = self.polynomial_func(xx, self.monomials)
            res.append(torch.mm(torch.mm(v.T, self.inverse_moments_matrix), v))
        return torch.stack(res).reshape(-1) 


    def __inv_score_samples_nquad__(self, *args):
        return 1 / self.score_samples(torch.tensor([[*args]]))[0]

    def __inv_score_samples__(self, x):
        return 1 / self.score_samples(x)

    def update(self, x, N):
        self.incr_func(self, x, N, self.inv_func)
        return self

    def save_model(self):
        return {
            "monomials": self.monomials,
            "moments_matrix": self.moments_matrix.tolist(),
            "inverse_moments_matrix": self.inverse_moments_matrix.tolist(),
        }

    def load_model(self, model_dict):
        self.monomials = model_dict["monomials"]
        self.moments_matrix = torch.tensor(model_dict["moments_matrix"], requires_grad=True)
        self.inverse_moments_matrix = torch.tensor(model_dict["inverse_moments_matrix"], requires_grad=True)
        return self

    def learned(self):
        return self.inverse_moments_matrix is not None

    def copy(self):
        mm_bis = MomentsMatrix(n=self.n)
        mm_bis.polynomial_func = self.polynomial_func
        mm_bis.incr_func = self.incr_func
        mm_bis.inv_func = self.inv_func
        mm_bis.monomials = self.monomials
        mm_bis.moments_matrix = self.moments_matrix
        mm_bis.inverse_moments_matrix = self.inverse_moments_matrix
        return mm_bis

""" Polynomials basis """

class PolynomialsBasis:
    @staticmethod
    def generate_combinations(max_degree, dimensions):
        def helper(remaining_dimensions, remaining_degree, combination):
            if remaining_dimensions == 0:
                combinations.append(combination)
                return
            for value in range(0, remaining_degree + 1):
                helper(remaining_dimensions - 1, remaining_degree - value, combination + [value])

        combinations = []
        helper(dimensions, max_degree, [])
        return sorted(combinations, key=lambda e: (torch.sum(torch.tensor(list(e))), list(-1 * torch.tensor(list(e)))))

    @staticmethod
    def apply_combinations(x, m, basis_func, device='cpu'):
        result = basis_func(x, m, device)
        return torch.prod(result, axis=1).reshape(-1, 1)

