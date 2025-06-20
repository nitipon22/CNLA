import torch
import torch.nn as nn

class CNLA(nn.Module):
    """
    CNLA: Classwise Normalization with Learnable Alignment.
    This module normalizes features (optionally by class) and aligns them with a learnable neural network.
    """

    def __init__(self, feature_dim, hidden_dim=64, classwise=False, num_classes=None, skip_aligner=False):
        super(CNLA, self).__init__()
        self.feature_dim = feature_dim
        self.classwise = classwise  # Whether to normalize per class
        self.num_classes = num_classes
        self.skip_aligner = skip_aligner  # If True, skip the aligner and return normalized features

        # Initialize normalization buffers
        if classwise and num_classes:
            # Per-class mean and std (size: [num_classes, feature_dim])
            self.register_buffer('mean_train', torch.zeros(num_classes, feature_dim))
            self.register_buffer('std_train', torch.ones(num_classes, feature_dim))
        else:
            # Global mean and std
            self.register_buffer('mean_train', torch.zeros(feature_dim))
            self.register_buffer('std_train', torch.ones(feature_dim))

        # Learnable aligner network (2-layer MLP)
        self.aligner = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    @staticmethod
    def coral_loss(source, target):
        """
        CORAL loss for feature alignment based on covariance matrices.
        Minimizes the difference in second-order statistics between source and target.
        """
        d = source.size(1)
        source_c = source - source.mean(dim=0)
        target_c = target - target.mean(dim=0)
        source_cov = (source_c.T @ source_c) / (source.size(0) - 1)
        target_cov = (target_c.T @ target_c) / (target.size(0) - 1)
        loss = torch.mean((source_cov - target_cov) ** 2)
        return loss / (4 * d * d)

    def set_distribution(self, features, labels=None):
        """
        Set the mean and std distribution using source training features.
        - If classwise is True, compute per-class stats.
        - Otherwise, compute global stats.
        """
        if self.classwise and labels is not None:
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    self.mean_train[c] = features[mask].mean(dim=0)
                    self.std_train[c] = features[mask].std(dim=0) + 1e-6
        else:
            self.mean_train = features.mean(dim=0)
            self.std_train = features.std(dim=0) + 1e-6

    def forward(self, x, y=None):
        """
        Normalize and align input feature x.
        - If classwise, use class-specific stats (y required).
        - Otherwise, normalize globally.
        - If skip_aligner is True, return normalized result directly.
        """
        if self.classwise and y is not None:
            print("Test labels passed to forward (classwise):", y)
            mean_batch = self.mean_train[y]
            std_batch = self.std_train[y]

            # Normalize test batch by its own mean/std
            x_norm = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

            # Rescale to source distribution
            x_aligned = x_norm * std_batch + mean_batch
        else:
            mean_batch = x.mean(dim=0)
            std_batch = x.std(dim=0) + 1e-6
            x_norm = (x - mean_batch) / std_batch
            x_aligned = x_norm * self.std_train + self.mean_train

        if self.skip_aligner:
            return x_aligned  # Skip alignment step if flag is set

        return self.aligner(x_aligned)  # Pass through learnable aligner

    def train_aligner(self, train_features, train_labels=None, epochs=1000, lr=1e-3, loss_type='mse', verbose=True):
        """
        Train the aligner network using source training features.
        Supported losses: 'mse' or 'coral'.
        """
        self.skip_aligner = False  # Make sure aligner is used
        optimizer = torch.optim.Adam(self.aligner.parameters(), lr=lr)
        criterion = nn.MSELoss() if loss_type == 'mse' else None

        self.train()  # Enable training mode

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward through aligner
            aligned = self.forward(train_features, train_labels)

            # Compute loss
            if loss_type == 'mse':
                loss = criterion(aligned, train_features)
            elif loss_type == 'coral':
                loss = self.coral_loss(aligned, train_features)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

            loss.backward()
            optimizer.step()

            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch == epochs - 1):
                print(f"[Aligner] Epoch {epoch + 1}/{epochs} - Loss ({loss_type}): {loss.item():.6f}")

        self.eval()  # Set to eval mode after training
        self.skip_aligner = False  # Keep aligner active for future use

    def debug_stats(self, features, labels=None, prefix=""):
        """
        Print mean and std of the current batch vs the stored training distribution.
        Useful for debugging normalization.
        """
        if self.classwise and labels is not None:
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    mean_batch = features[mask].mean(dim=0).cpu().numpy()
                    std_batch = features[mask].std(dim=0).cpu().numpy()
                    mean_train = self.mean_train[c].cpu().numpy()
                    std_train = self.std_train[c].cpu().numpy()
                    print(f"{prefix}Train mean vector per class {c}: {mean_train}")
                    print(f"{prefix}Train std vector per class {c}: {std_train}")
                    print(f"{prefix}Test batch mean for class {c}: {mean_batch}")
                    print(f"{prefix}Test batch std  for class {c}: {std_batch}")
        else:
            mean_batch = features.mean(dim=0).cpu().numpy()
            std_batch = features.std(dim=0).cpu().numpy()
            mean_train = self.mean_train.cpu().numpy()
            std_train = self.std_train.cpu().numpy()
            print(f"{prefix}Train mean vector: {mean_train}")
            print(f"{prefix}Train std vector: {std_train}")
            print(f"{prefix}Test batch mean: {mean_batch}")
            print(f"{prefix}Test batch std: {std_batch}")

    def pseudo_labeling_alignment(self, clf, target_features, num_pseudo_rounds=5):
        """
        Perform iterative pseudo-labeling and alignment on unlabeled target features.
        Steps:
        - Predict pseudo-labels using given classifier.
        - Align using CNLA with pseudo-labels.
        - Refit classifier on aligned features.
        """
        X_test = target_features.copy()

        for round in range(num_pseudo_rounds):
            print(f"Pseudo-labeling round {round + 1}/{num_pseudo_rounds}")

            # Predict pseudo-labels
            pseudo_labels = clf.predict(X_test)
            pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)

            # Align using CNLA
            test_features_tensor = torch.tensor(X_test, dtype=torch.float32)
            aligned_test_tensor = self.forward(test_features_tensor, y=pseudo_labels_tensor).detach()
            X_test_aligned_np = aligned_test_tensor.cpu().numpy()

            # Re-train classifier on aligned + pseudo-labeled data
            clf.fit(X_test_aligned_np, pseudo_labels)

            # Update test features for next round
            X_test = X_test_aligned_np

        return X_test  # Return final aligned features
