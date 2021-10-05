import torch 
import numpy as np

class RademacherCovariates:

    def generate_covariates_matrix(
            n, p):
        dtype = torch.float32
        mean = np.zeros(p)
        samples = np.random.binomial(1, 0.5, (n, p)) * 2 - 1
        samples = samples
        samples = torch.tensor(samples, dtype=dtype)
        return samples


class LinearRegressionDataset:

    def __init__(self, p, noise_std, beta):

        self.p = p
        self.noise_std = noise_std

        self.beta = beta
        self.covariates = RademacherCovariates

    def generate_data(self, n):
        dtype = torch.float32
        beta = torch.tensor(self.beta, dtype=dtype)
        p = self.p
        X = self.covariates.generate_covariates_matrix(n, p)
        y = torch.mm(X, beta)
        noise = self.noise_std * torch.randn(n, 1, dtype=dtype)
        y += noise
        return X, y, noise

def get_measurement(one_pt, n=100, seed=42):
    one_pt = one_pt.reshape(-1)/255
    one_pt = one_pt.unsqueeze(-1)
    p = one_pt.size(0)
    np.random.seed(seed)
    X = torch.tensor(np.random.binomial(1, 0.25, (n,p))*2 - 1,
                    dtype = torch.float32)
    y = torch.mm(X, one_pt)

    return y, X