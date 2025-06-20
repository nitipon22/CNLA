import torch
import torch.nn.functional as nn

def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute the Gaussian Kernel matrix between source (x) and target (y).

    Args:
        x: Tensor of shape (n_samples_x, n_features)
        y: Tensor of shape (n_samples_y, n_features)
        kernel_mul: Multiplicative factor for bandwidth scaling
        kernel_num: Number of different Gaussian kernels to use (multi-scale)
        fix_sigma: Optional fixed bandwidth (if None, it will be computed)

    Returns:
        A combined multi-scale Gaussian kernel matrix of shape (n_total, n_total)
    """
    n_samples = int(x.size(0)) + int(y.size(0))
    total = torch.cat([x, y], dim=0)

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)

    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

    return sum(kernel_val)  # Combine multiple kernels

def mmd_loss(source, target):
    """
    Compute the Maximum Mean Discrepancy (MMD) loss between source and target.

    Args:
        source: Source domain features (batch_size, feature_dim)
        target: Target domain features (batch_size, feature_dim)

    Returns:
        MMD loss (scalar)
    """
    batch_size = source.size(0)
    kernels = gaussian_kernel(source, target)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)
    return loss

class DomainDiscriminator(nn.Module):
    """
    Simple domain discriminator used for adversarial domain adaptation.

    Architecture: input → hidden → ReLU → output (1 unit)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(DomainDiscriminator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def coral_loss(source, target):
    """
    Compute the CORAL (Correlation Alignment) loss between source and target.

    Args:
        source: Source domain features (batch_size, feature_dim)
        target: Target domain features (batch_size, feature_dim)

    Returns:
        CORAL loss (scalar)
    """
    d = source.size(1)

    source_c = source - source.mean(dim=0)
    target_c = target - target.mean(dim=0)

    source_cov = (source_c.T @ source_c) / (source.size(0) - 1)
    target_cov = (target_c.T @ target_c) / (target.size(0) - 1)

    loss = torch.mean((source_cov - target_cov) ** 2)
    return loss / (4 * d * d)
