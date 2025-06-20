# Import necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.decomposition import PCA
import pandas as pd
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from CNla.core import CNLA  # Assumed to be available

# --- Dataset Loaders (including URL-based datasets) ---

def load_pima():
    # Load Pima Indians Diabetes dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
            "BMI","DiabetesPedigreeFunction","Age","Outcome"]
    df = pd.read_csv(url, names=cols)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def load_glass():
    # Load Glass Identification dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv"
    cols = ["Id","RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"]
    df = pd.read_csv(url, names=cols, skiprows=1)
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values - 1
    return X, y

def load_wine_quality():
    # Load Wine Quality (Red) dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    X = df.iloc[:, :-1].values
    y = df['quality'].values
    return X, y

def bin_wine_quality(y):
    # Convert wine quality scores into 3 classes: low, medium, high
    y_binned = np.zeros_like(y)
    y_binned[y <= 5] = 0
    y_binned[y == 6] = 1
    y_binned[y >= 7] = 2
    return y_binned

# Mapping of dataset names to loading functions
dataset_loaders = {
    'iris': load_iris,
    'wine': load_wine,
    'breast_cancer': load_breast_cancer,
    'digits': load_digits,
    'pima_diabetes': load_pima,
    # 'glass': load_glass,  # Uncomment if needed
    'wine_quality': load_wine_quality,
}

results = []  # To store evaluation results

# --- Visualization Function ---
def plot_decision_boundary(clf, X, y, title="Decision Boundary"):
    # Plot decision boundary in 2D
    h = 0.1
    x_mean, x_std = X[:, 0].mean(), X[:, 0].std()
    y_mean, y_std = X[:, 1].mean(), X[:, 1].std()
    x_min, x_max = x_mean - 2 * x_std, x_mean + 2 * x_std
    y_min, y_max = y_mean - 2 * y_std, y_mean + 2 * y_std

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    try:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(grid_points)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

        for cls in np.unique(y):
            plt.scatter(X[y == cls, 0], X[y == cls, 1], label=f'Class {cls}', edgecolor='k')
        plt.title(title)
        plt.legend()
        plt.show()
    except MemoryError:
        print("⚠️ MemoryError: Mesh grid too large. Try increasing `h` or reducing axis range.")

# --- Main Evaluation Loop over Each Dataset ---

for name, loader in dataset_loaders.items():
    print(f"\n--- Dataset: {name} ---")
    data = loader()

    # Support both sklearn and tuple-style datasets
    if hasattr(data, 'data') and hasattr(data, 'target'):
        X, y = data.data, data.target
    else:
        X, y = data

    if name == 'wine_quality':
        y = bin_wine_quality(y)

    # Remove NaNs
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    # Use Stratified K-Fold Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize metric lists
    accs_before, accs_after = [], []
    f1s_before, f1s_after = [], []
    f1m_before, f1m_after = [], []

    per_fold_result = []  # Per-fold logging

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f" Fold {fold} / 5")
        X_train, X_test_orig = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_test = X_test_orig.copy()

        # Artificial distribution shift on some features
        if X.shape[1] > 3:
            X_test[:, 1] = X_test[:, 1] * 1.5 + 0.2
            X_test[:, 3] = X_test[:, 3] * 0.7 - 0.1

        # Convert to torch tensors
        feature_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        train_features = torch.tensor(X_train, dtype=torch.float32)
        train_labels = torch.tensor(y_train, dtype=torch.long)
        test_features = torch.tensor(X_test, dtype=torch.float32)
        test_labels = torch.tensor(y_test, dtype=torch.long)

        # 1. Initialize CNLA model
        model = CNLA(feature_dim=feature_dim, classwise=True, num_classes=num_classes)
        model.set_distribution(train_features, train_labels)

        # 2. Train learnable aligner
        model.skip_aligner = False
        model.train_aligner(train_features, train_labels, epochs=1000, lr=1e-3)

        # 3. Train classifier on unaligned source data
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Keep original for pre-alignment prediction
        X_test_original = X_test.copy()

        # 4. Align test set using pseudo-labels
        aligned_target_features = model.pseudo_labeling_alignment(clf, X_test, num_pseudo_rounds=5)

        # 5. Get final pseudo-labels
        pseudo_labels_final = clf.predict(aligned_target_features)
        pseudo_labels_tensor = torch.tensor(pseudo_labels_final, dtype=torch.long)

        # 6. Final alignment
        model.eval()
        test_features_tensor = torch.tensor(X_test, dtype=torch.float32)
        aligned_test_tensor = model(test_features_tensor, y=pseudo_labels_tensor).detach()

        # 7. Predict before alignment
        y_pred_before = clf.predict(X_test_original)

        # 8. Predict after alignment
        X_test_aligned_np = aligned_test_tensor.cpu().numpy()
        y_pred_after = clf.predict(X_test_aligned_np)

        # Compute metrics
        acc_before = accuracy_score(y_test, y_pred_before)
        acc_after = accuracy_score(y_test, y_pred_after)
        f1_micro_before = f1_score(y_test, y_pred_before, average='micro')
        f1_micro_after = f1_score(y_test, y_pred_after, average='micro')
        f1_before = f1_score(y_test, y_pred_before, average='macro')
        f1_after = f1_score(y_test, y_pred_after, average='macro')

        accs_before.append(acc_before)
        accs_after.append(acc_after)
        f1m_before.append(f1_micro_before)
        f1m_after.append(f1_micro_after)
        f1s_before.append(f1_before)
        f1s_after.append(f1_after)

        # Store per fold results
        per_fold_result.append({
            'Fold': fold,
            'Accuracy Before': acc_before,
            'Accuracy After': acc_after,
            'F1-macro Before': f1_before,
            'F1-macro After': f1_after,
            'F1-micro Before': f1_micro_before,
            'F1-micro After': f1_micro_after,
        })

    # Aggregate statistics
    mean_acc_before = np.mean(accs_before)
    mean_acc_after = np.mean(accs_after)
    mean_f1_before = np.mean(f1s_before)
    mean_f1_after = np.mean(f1s_after)
    mean_f1m_before = np.mean(f1m_before)
    mean_f1m_after = np.mean(f1m_after)
    std_acc_before = np.std(accs_before)
    std_acc_after = np.std(accs_after)
    std_f1_before = np.std(f1s_before)
    std_f1_after = np.std(f1s_after)
    std_f1m_before = np.std(f1m_before)
    std_f1m_after = np.std(f1m_after)

    print(f"\nDataset: {name} 5-fold mean accuracy before CNLA: {mean_acc_before:.4f}")
    print(f"Dataset: {name} 5-fold mean accuracy after CNLA:  {mean_acc_after:.4f}")
    print(f"Dataset: {name} 5-fold mean F1 Macro before CNLA: {mean_f1_before:.4f}")
    print(f"Dataset: {name} 5-fold mean F1 Macro after CNLA:  {mean_f1_after:.4f}")

    # Store overall results
    results.append({
        'dataset': name,
        'acc_before_mean': mean_acc_before,
        'acc_after_mean': mean_acc_after,
        'f1_before_mean': mean_f1_before,
        'f1_after_mean': mean_f1_after,
        'f1m_before_mean': mean_f1m_before,
        'f1m_after_mean': mean_f1m_after,
        'acc_before_std': std_acc_before,
        'acc_after_std': std_acc_after,
        'f1_before_std': std_f1_before,
        'f1_after_std': std_f1_after,
        'f1m_before_std': std_f1m_before,
        'f1m_after_std': std_f1m_after,
        'fold_results': per_fold_result
    })

    # --- Visualization with PCA ---
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)
    clf_pca = LogisticRegression(max_iter=1000, random_state=42)
    clf_pca.fit(X_test_pca, y_test)

    plot_decision_boundary(clf_pca, X_test_pca, pseudo_labels_final, title=f"{name} - Pseudo-labels PCA Decision Boundary")

    # Plot aligned features after PCA transformation
    X_test_aligned_pca = pca.transform(aligned_test_tensor.numpy())
    plot_decision_boundary(clf_pca, X_test_aligned_pca, pseudo_labels_final, title=f"{name} - Aligned Pseudo PCA Decision Boundary")

# --- Final Summary ---
print("\nSummary Results:")
for r in results:
    print(f"Dataset: {r['dataset']}")
    print(f" Accuracy Before: {r['acc_before_mean']:.4f} ± {r['acc_before_std']:.4f}")
    print(f" Accuracy After:  {r['acc_after_mean']:.4f} ± {r['acc_after_std']:.4f}")
    print(f" F1-score Before: {r['f1_before_mean']:.4f} ± {r['f1_before_std']:.4f}")
    print(f" F1-score After:  {r['f1_after_mean']:.4f} ± {r['f1_after_std']:.4f}")
    print(f" F1-micro Before: {r['f1m_before_mean']:.4f} ± {r['f1m_before_std']:.4f}")
    print(f" F1-micro After:  {r['f1m_after_mean']:.4f} ± {r['f1m_after_std']:.4f}")
    print("Per fold results:")
    for fold_res in r['fold_results']:
        print(f" Fold {fold_res['Fold']:>2}: "
              f"Acc Before={fold_res['Accuracy Before']:.4f}, "
              f"Acc After={fold_res['Accuracy After']:.4f}, "
              f"F1-macro Before={fold_res['F1-macro Before']:.4f}, "
              f"F1-macro After={fold_res['F1-macro After']:.4f}, "
              f"F1-micro Before={fold_res['F1-micro Before']:.4f}, "
              f"F1-micro After={fold_res['F1-micro After']:.4f}")
    print()
