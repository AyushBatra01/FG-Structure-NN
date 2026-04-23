import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FactorGraphNetwork(nn.Module):
    def __init__(self, n_vars, alphabet_size, K=None, hidden_dims=(16,16), max_factor_size=None, shared_mlp=False, seed=None):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.n_vars = n_vars
        self.alphabet_size = alphabet_size
        self.K = n_vars if K is None else K
        self.max_factor_size = max_factor_size
        self.shared_mlp = shared_mlp
        
        # specify input dimension; just = number variables if discrete binary or continuous
        if alphabet_size <= 2:
            self.input_dim = n_vars
            self.one_hot = False
        else:
            self.input_dim = n_vars * alphabet_size
            self.one_hot = True

        # variable masks
        self.raw_masks = nn.Parameter(0.1 * torch.randn(K, n_vars))

        # input to factor Multilayer Perceptrons
        def build_mlp(in_dim, hidden_dims):
            layers = []
            prev = in_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        if shared_mlp:
            self.mlp = build_mlp(self.input_dim, hidden_dims)
        else:
            self.mlps = nn.ModuleList(
                [build_mlp(self.input_dim, hidden_dims) for _ in range(K)]
            )

        # factor weights
        self.raw_weights = nn.Parameter(0.1 * torch.randn(K))

    def soft_masks(self):
        return torch.sigmoid(self.raw_masks)

    def soft_weights(self):
        return torch.sigmoid(self.raw_weights)

    def masks(self, threshold=0.3):
        # soft mask
        mask_soft = self.soft_masks()  # (K, n_vars)
        # hard mask (0 or 1)
        if self.max_factor_size is not None:
            K, n_vars = mask_soft.shape
            k = min(self.max_factor_size, n_vars)
            # get top-k indices per factor
            topk_vals, topk_idx = torch.topk(mask_soft, k=k, dim=1)
            # build top-k mask
            topk_mask = torch.zeros_like(mask_soft)
            topk_mask.scatter_(1, topk_idx, 1.0)
            # threshold condition
            thresh_mask = (mask_soft > threshold).float()
            # combine both conditions
            mask_hard = topk_mask * thresh_mask
        else:
            # no max factor size
            mask_hard = (mask_soft > threshold).float()
        # straight-through estimator
        mask = mask_hard.detach() + mask_soft - mask_soft.detach()
        return mask

    def weights(self, threshold=0.3):
        # soft mask
        weight_soft = self.soft_weights()
        # hard mask (0 or 1)
        weight_hard = (weight_soft > threshold).float()
        # straight-through estimator
        weight = weight_hard.detach() + weight_soft - weight_soft.detach()
        return weight

    def _apply_mask(self, x: torch.Tensor, mask_k: torch.Tensor) -> torch.Tensor:
        if self.one_hot:
            mask_expanded = mask_k.repeat_interleave(self.alphabet_size)
        else:
            mask_expanded = mask_k
        return x * mask_expanded.unsqueeze(0)

    def forward(self, x):
        masks = self.masks() 
        weights = self.weights()

        factor_values = []  
        for k in range(self.K):
            x_masked = self._apply_mask(x, masks[k])
            mlp = self.mlp if self.shared_mlp else self.mlps[k]
            h_k = mlp(x_masked).squeeze(-1)            # (batch,)
            factor_values.append(h_k)

        # Stack: (batch, K)
        H = torch.stack(factor_values, dim=1)
        # Stage 2: weighted sum  ->  (batch,)
        output = (H * weights.unsqueeze(0)).sum(dim=1)
        return output

    def regularization_loss(self, lambda_mask=0.1, lambda_weight=0.1, lambda_mlp_l2=0, lambda_bp=0):
        masks = self.masks()
        weights = self.weights()
        # mask penalty
        mask_reg = masks.mean()
        # weight penalty
        weight_reg = weights.mean()
        # L2 MLP weight penalty
        mlp_l2 = 0.0
        count = 0
        for p in self.parameters():
            if p.requires_grad and p is not self.raw_masks and p is not self.raw_weights:
                mlp_l2 += (p**2).sum()  
                count += p.numel()
        mlp_l2 /= count
        # Message passing penalty
        # soft_masks = self.soft_masks()
        # soft_weights = self.soft_weights()
        # sizes = soft_masks.sum(dim=1)
        # log_q = np.log(self.alphabet_size) # for stability
        # msg_cost = torch.exp(log_q * sizes)
        # msg_penalty = (soft_weights * sizes * msg_cost).mean()
        sizes = masks.sum(dim=1)
        log_q = np.log(self.alphabet_size)
        baseline = torch.exp(log_q * torch.tensor(2))  # cost of size-2 factor
        # msg_cost = torch.exp(log_q * sizes)
        msg_cost = torch.clamp(torch.exp(log_q * sizes) - baseline, min=0.0)
        msg_penalty = (weights * sizes * msg_cost).mean()
        return lambda_mask * mask_reg + lambda_weight * weight_reg + lambda_mlp_l2 * mlp_l2 + lambda_bp * msg_penalty

    def avg_factor_size(self):
        weights = self.weights()
        masks = self.masks()
        factor_sizes = masks.sum(dim=1)
        active = (weights == 1) & (factor_sizes > 0)
        if active.sum() == 0:
            return torch.tensor(0.0, device=weights.device)
        return factor_sizes[active].mean()

    def n_active_factors(self):
        weights = self.weights()
        masks = self.masks()
        factor_sizes = masks.sum(dim=1)
        active = (weights == 1) & (factor_sizes > 0)
        return active.sum()