import torch
import numpy as np
from utils import LinearRegressionDataset


class Dataset:
    def __init__(self, n=80, p=200, k=5, noise_std=.5, beta=None, random_state=None):
        self.n = n
        self.p = p
        self.k = k
        self.noise_std = noise_std
        self.beta = beta

        self.dataset = LinearRegressionDataset(
            p=p, noise_std=noise_std, beta=beta)
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
        self.X, self.y, self.noise = self.dataset.generate_data(n)


class ImplicitN(torch.nn.Module):
    def __init__(self, p, N, if_positive=False):
        super().__init__()
        self.N = N
        self.if_positive = if_positive
        self.layer = torch.nn.Linear(p, 1, bias=False)
        self.layer2 = torch.nn.Linear(p, 1, bias=False)

    def forward(self, x):
        if self.if_positive:
            recovered_w = self.layer.weight ** self.N
        else:
            recovered_w = self.layer.weight ** self.N - self.layer2.weight ** self.N
        y = x.matmul(recovered_w.t())
        return y

    def get_loss_criterion(self):
        return torch.nn.MSELoss()

    def get_params(self):
        if self.if_positive:
            return self.layer.weight ** self.N
        else:
            return self.layer.weight ** self.N - self.layer2.weight ** self.N

    def init_weights(self, alpha):
        torch.nn.init.ones_(self.layer.weight)
        torch.nn.init.ones_(self.layer2.weight)

        self.layer.weight.data *= alpha
        self.layer2.weight.data *= alpha


class Trainer:
    def __init__(self, X, y, lr, model):
        self.X = X
        self.y = y
        self.lr = lr
        self.model = model

    def do_one_epoch(self):
        n = self.X.size()[0]
        self.model.train()
        X = self.X
        y = self.y
        y_pred = self.model(X)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        loss_criterion = self.model.get_loss_criterion()
        loss = loss_criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class Simulation:
    def __init__(self, dataset, N, w0, lr, epochs, if_positive=True):
        self.dataset = dataset
        self.N = N
        self.w0 = w0
        self.lr = lr
        self.epochs = epochs
        self.alpha = w0**(1/N)

        self.l2_squared_errors = []
        self.recovered_path = []

        self.p = dataset.p
        self.k = dataset.k
        self.n = dataset.n
        self.noise_std = dataset.noise_std

        self.model = ImplicitN(p=dataset.p, N=N, if_positive=if_positive)
        self.model.init_weights(w0**(1/N))
        self.trainer = Trainer(X=dataset.X, y=dataset.y,
                               lr=lr, model=self.model)
        self.beta = torch.tensor(dataset.beta, dtype=torch.float32)

        self.l2_squared_errors = []
        self.recovered_path = []

        self.compute_ls()

    def compute_ls(self):
        X = self.dataset.X.detach().numpy()
        y = self.dataset.y.detach().numpy()
        self.beta_ls = np.linalg.lstsq(X, y, rcond=-1)[0].flatten()

    def _one_epoch_metric(self):
        recovered_w = self.model.get_params()
        self.recovered_path.append(recovered_w.detach().numpy())

        l2_squared_error = np.sum(
            (recovered_w.detach().numpy().flatten() - self.beta.numpy().flatten())**2)
        self.l2_squared_errors.append(l2_squared_error)
        self.trainer.do_one_epoch()


    def train(self, epochs=None):
        if epochs is None:
            epochs = self.epochs
        for i in range(epochs):
            self._one_epoch_metric()
        self.format_metrics()

    def format_metrics(self):
        path_mat = np.vstack(self.recovered_path)

        self.signal = path_mat[:, range(self.k)]
        noise = path_mat[:, range(self.k, self.p)]

        self.noise_m = np.mean(noise, axis=1)
        self.noise_s = np.std(noise, axis=1)
        self.noise_max = np.max(noise, axis=1)

class ExpMNIST:
    def __init__(self, N, w0, lr, epochs, 
                 X, y, one_pt, if_positive=True):
        self.X = X
        self.y = y
        
        self.N = N
        self.w0 = w0
        self.lr = lr
        self.epochs = epochs
        
        n = self.X.size(0)
        p = self.X.size(1)
        self.model = ImplicitN(p=p, N=N, if_positive=if_positive)
        self.model.init_weights(w0**(1/N))
        self.trainer = Trainer(X=X, y=y, lr=lr, model=self.model)
        self.beta = one_pt.reshape(-1)/255
        
        self.l2_squared_errors = []
        self.compute_ls()

    def compute_ls(self):
        X = self.X.detach().numpy()
        y = self.y.detach().numpy()
        self.beta_ls = np.linalg.lstsq(X, y, rcond=-1)[0].flatten()

    def _one_epoch_metric(self):
        recovered_w = self.model.get_params()

        l2_squared_error = np.sum(
            (recovered_w.detach().numpy().flatten() - self.beta.numpy().flatten())**2)
        self.l2_squared_errors.append(l2_squared_error)
        self.trainer.do_one_epoch()

        return l2_squared_error

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.epochs
        prev_error = float('inf')
        cnt = 0
        for i in range(epochs):
            cur_error = self._one_epoch_metric()        
            if cnt == 0 and cur_error > prev_error:
                recovered_w = self.model.get_params().detach()
                cnt = 1
            prev_error = cur_error
#         return recovered_w
