import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def one_hot_encode(data: np.ndarray, alphabet_size: int) -> np.ndarray:
    """
    Parameters
    ----------
    data : (N, n_vars) integer array
    alphabet_size : int

    Returns
    -------
    (N, n_vars * alphabet_size) float32 array
    """
    N, n_vars = data.shape
    out = np.zeros((N, n_vars * alphabet_size), dtype=np.float32)
    for i in range(n_vars):
        out[np.arange(N), i * alphabet_size + data[:, i]] = 1.0
    return out

def to_network_input(data: np.ndarray, one_hot: bool, alphabet_size: int) -> torch.Tensor:
    """Convert integer-coded data array to the float tensor the network expects."""
    if one_hot:
        arr = one_hot_encode(data, alphabet_size)
    else:
        arr = data.astype(np.float32)
    return torch.tensor(arr, dtype=torch.float32)



class FactorGraphLearner:
    def __init__(self, n_vars, alphabet_size, K, noise_generator, hidden_dims=(16,16), max_factor_size=None, shared_mlp=False, seed=None):
        self.network = FactorGraphNetwork(n_vars, alphabet_size, K, hidden_dims, max_factor_size, shared_mlp, seed=seed)
        self.noise_generator = noise_generator

    def nce_loss(self, x_real, x_fake, log_q_real, log_q_fake):
        model = self.network
        s_real = model(x_real) - log_q_real
        s_fake = model(x_fake) - log_q_fake
        logits = torch.cat([s_real, s_fake], dim=0)
        labels = torch.cat([torch.ones_like(s_real), torch.zeros_like(s_fake)], dim=0)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def scale_penalty(self, base_lambda, epoch, max_epochs, schedule=True, start=0.1, end=0.5):
        # linear scaling of lambda penalty
        if not schedule:
            return base_lambda
        if start <= 1:
            start = int(start * max_epochs)
        if end <= 1:
            end = int(end * max_epochs)
        if epoch <= start:
            return 0
        elif epoch > end:
            return base_lambda
        else:
            return base_lambda * (epoch - start) / (end - start)

    def scale_all_penalties(self, epoch, max_epochs, lambda_mask, lambda_weight, lambda_bp, schedule=True, start=0.1, end=0.5):
        l_mask = self.scale_penalty(lambda_mask, epoch, max_epochs, schedule, start, end)
        l_wt = self.scale_penalty(lambda_weight, epoch, max_epochs, schedule, start, end)
        l_bp = self.scale_penalty(lambda_bp, epoch, max_epochs, schedule, start, end)
        return l_mask, l_wt, l_bp
        

    def train(self, data, n_epochs = 1000, batch_size = 128, lr = 1e-3, 
              lambda_mask = 0.1, lambda_weight = 0.1, lambda_mlp_l2 = 0.01, lambda_bp = 0.01, penalty_schedule=False,
              noise_ratio = 1, device = "cpu", verbose = True, log_every = 50):
        # train network using NCE
        model = self.network.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # real data - one hot encodings and log q
        x_real = to_network_input(data, model.one_hot, model.alphabet_size).to(device)
        log_q_real = torch.tensor(self.noise_generator.log_prob(data), dtype=torch.float32)

        # dataset
        dataset = TensorDataset(x_real, log_q_real)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = {x : [] for x in ['Total', 'Main', 'Reg']}
        diagnostics = {x : [] for x in ["avg_factor_size", "n_active_factors"]}
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_loss_main = 0.0
            epoch_loss_reg = 0.0
            for x_batch, lq_batch in loader:
                x_batch = x_batch.to(device)
                lq_batch = lq_batch.to(device)
                bn = x_batch.shape[0]

                # sample fake data
                x_fake = self.noise_generator.generate_samples(n_samples=int(noise_ratio*bn))
                lq_fake = self.noise_generator.log_prob(x_fake)
                x_fake = to_network_input(x_fake, model.one_hot, model.alphabet_size).to(device)
                lq_fake = torch.tensor(lq_fake, dtype=torch.float32).to(device)

                # train step
                optimizer.zero_grad()
                main_loss = self.nce_loss(x_batch, x_fake, lq_batch, lq_fake)
                l_mask, l_wt, l_bp = self.scale_all_penalties(epoch, n_epochs, lambda_mask, lambda_weight, lambda_bp, penalty_schedule)
                reg_loss = self.network.regularization_loss(l_mask, l_wt, lambda_mlp_l2, l_bp)
                # reg_loss = self.network.regularization_loss(lambda_mask, lambda_weight, lambda_mlp_l2, lambda_bp)
                loss = main_loss + reg_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bn
                epoch_loss_main += main_loss.item() * bn
                epoch_loss_reg += reg_loss.item() * bn
    
            avg_loss = epoch_loss / len(data)
            avg_loss_main = epoch_loss_main / len(data)
            avg_loss_reg = epoch_loss_reg / len(data)
            for key, val in [('Total', avg_loss), ('Main', avg_loss_main), ('Reg', avg_loss_reg)]:
                losses[key].append(val)
            if verbose and (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1:4d}/{n_epochs}  loss={avg_loss:.4f}  main={avg_loss_main:.4f}  reg={avg_loss_reg:.4f}")

            avg_factor_size = self.network.avg_factor_size().detach().cpu().numpy()
            n_active_factors = self.network.n_active_factors().detach().cpu().numpy()
            diagnostics["avg_factor_size"].append(avg_factor_size)
            diagnostics["n_active_factors"].append(n_active_factors)
        return losses, diagnostics

    def extract_graph(self, mask_thresh=0.5, weight_thresh=0.5):
        self.network.eval()
        masks = self.network.masks().detach().cpu().numpy()  
        weights = self.network.weights().detach().cpu().numpy()  
        self.network.train()
        active = [
            k for k in range(self.network.K) if weights[k] > weight_thresh and np.any(masks[k] > mask_thresh)
        ]
        scopes = {
            k: [i for i in range(self.network.n_vars) if masks[k, i] > mask_thresh] for k in active
        }
        return {
            "active_factors": active,
            "factor_scopes": scopes,
            "mask_values": masks,
            "weight_values":  weights,
        }

    def predict(self, data):
        self.network.eval()
        x_t = to_network_input(data, self.network.one_hot, self.network.alphabet_size)
        with torch.no_grad():
            out = self.network(x_t)
        self.network.train()
        return out.cpu().numpy()