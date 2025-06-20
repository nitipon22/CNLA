# === Import Libraries ===
import numpy as np
import torch
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.linalg import sqrtm, inv
from sklearn.datasets import fetch_openml

# === Loaders for specific datasets ===

def load_mnist():
    # Load MNIST digits dataset (handwritten digits)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    return X, y

def load_usps():
    # Load USPS dataset (handwritten digits, similar to MNIST)
    usps = fetch_openml('usps', version=1, as_frame=False)
    X = usps.data / 255.0
    y = usps.target.astype(int)
    return X, y

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
    # Load Wine Quality dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    X = df.iloc[:, :-1].values
    y = df['quality'].values
    return X, y

def bin_wine_quality(y):
    # Convert wine quality scores to 3-class labels: 0 (low), 1 (medium), 2 (high)
    y_binned = np.zeros_like(y)
    y_binned[y <= 5] = 0
    y_binned[y == 6] = 1
    y_binned[y >= 7] = 2
    return y_binned

# === Alignment Methods ===

def align_standard_no_ytest(X_src, y_src, X_tgt):
    # Apply standard normalization (mean/std) from source to target globally
    scaler = StandardScaler().fit(X_src)
    return scaler.transform(X_tgt)

def align_zscore_classwise(X_src, y_src, X_tgt, num_pseudo_rounds=5):
    """
    Classwise Z-score normalization using pseudo-labeling.
    - Train classifier on source data.
    - Predict pseudo-labels on target.
    - Normalize target by class-wise source stats.
    - Repeat for several rounds.
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_src, y_src)
    X_test = X_tgt.copy()

    for round in range(num_pseudo_rounds):
        print(f"Pseudo-labeling round {round + 1}/{num_pseudo_rounds}")
        y_pseudo = clf.predict(X_test)
        X_test_aligned = np.zeros_like(X_test)

        for cls in np.unique(y_src):
            X_cls = X_src[y_src == cls]
            mean = X_cls.mean(axis=0)
            std = X_cls.std(axis=0) + 1e-8

            idx = np.where(y_pseudo == cls)[0]
            if len(idx) > 0:
                X_test_aligned[idx] = (X_test[idx] - mean) / std

        unique_classes = np.unique(y_pseudo)
        if len(unique_classes) > 1:
            clf.fit(X_test_aligned, y_pseudo)
            X_test = X_test_aligned
        else:
            print(f"Warning: Only one class predicted at round {round+1}. Stopping iteration.")
            return X_test_aligned

    # Final alignment using last pseudo-labels
    y_pseudo_final = clf.predict(X_test)
    X_test_final_aligned = np.zeros_like(X_test)

    for cls in np.unique(y_src):
        X_cls = X_src[y_src == cls]
        mean = X_cls.mean(axis=0)
        std = X_cls.std(axis=0) + 1e-8

        idx = np.where(y_pseudo_final == cls)[0]
        if len(idx) > 0:
            X_test_final_aligned[idx] = (X_test[idx] - mean) / std

    return X_test_final_aligned

def align_batchnorm_no_ytest(X_src, y_src, X_tgt):
    # BatchNorm-style normalization (mean/std of target itself, not source)
    mean = X_tgt.mean(axis=0)
    std = X_tgt.std(axis=0) + 1e-8
    return (X_tgt - mean) / std

def align_coral_no_ytest(X_src, y_src, X_tgt):
    # CORAL: Global alignment using covariance matrices (unsupervised)
    reg = 1e-6
    cov_s = np.cov(X_src, rowvar=False) + reg * np.eye(X_src.shape[1])
    cov_t = np.cov(X_tgt, rowvar=False) + reg * np.eye(X_tgt.shape[1])
    A = sqrtm(cov_s)
    B = sqrtm(cov_t)
    if np.iscomplexobj(A): A = A.real
    if np.iscomplexobj(B): B = B.real
    return X_tgt @ inv(B) @ A

# === Summary function for datasets ===

def print_dataset_summary(X, y, dataset_name="Dataset"):
    print(f"--- Summary of {dataset_name} ---")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    classes, counts = np.unique(y, return_counts=True)
    df_dist = pd.DataFrame({'Class': classes, 'Count': counts})
    df_dist['Percentage'] = 100 * df_dist['Count'] / df_dist['Count'].sum()
    print(df_dist.to_string(index=False))
    print()

# === Dataset + Method Config ===

dataset_loaders = {
    'iris': load_iris,
    'wine': load_wine,
    'breast_cancer': load_breast_cancer,
    'digits': load_digits,
    'pima_diabetes': load_pima,
    'wine_quality': load_wine_quality,
    # 'mnist': load_mnist,
    # 'usps': load_usps,
}

methods = {
    'StandardScaler': align_standard_no_ytest,
    'Zscore': align_zscore_classwise,
    'BatchNorm': align_batchnorm_no_ytest,
    'CORAL': align_coral_no_ytest,
}

# === Run Experiments ===

results = []

for name, loader in dataset_loaders.items():
    print(f"\n--- Dataset: {name} ---")
    data = loader()
    if hasattr(data, 'data') and hasattr(data, 'target'):
        X, y = data.data, data.target
    else:
        X, y = data

    if name == 'wine_quality':
        y = bin_wine_quality(y)

    # Remove missing labels if any
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    print_dataset_summary(X, y, dataset_name=name)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train base classifier
        clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
        clf.fit(X_train, y_train)

        for mname, func in methods.items():
            # Apply alignment method
            X_test_aligned = func(X_train, y_train, X_test)

            # Sanitize values (remove NaNs/infs)
            X_test_aligned = np.nan_to_num(X_test_aligned, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test_aligned = np.clip(X_test_aligned, -1e6, 1e6)

            # Predict and evaluate
            y_pred = clf.predict(X_test_aligned)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            f1_micro = f1_score(y_test, y_pred, average='micro')

            # Log result
            results.append({
                'Dataset': name,
                'Method': mname,
                'Fold': fold,
                'Accuracy': acc,
                'F1_macro': f1,
                'F1_micro': f1_micro
            })

# === Summarize Results ===

df = pd.DataFrame(results)
summary = df.groupby(['Dataset', 'Method']).agg({
    'Accuracy': ['mean', 'std'],
    'F1_macro': ['mean', 'std'],
    'F1_micro': ['mean', 'std']
}).reset_index()

# Flatten MultiIndex columns
summary.columns = ['Dataset', 'Method',
                   'Acc_Mean', 'Acc_Std',
                   'F1_macro_Mean', 'F1_macro_Std',
                   'F1_micro_Mean', 'F1_micro_Std']

# Format for display
summary['Acc_Mean±Std'] = summary.apply(lambda x: f"{x['Acc_Mean']:.3f} ± {x['Acc_Std']:.3f}", axis=1)
summary['F1_macro_Mean±Std'] = summary.apply(lambda x: f"{x['F1_macro_Mean']:.3f} ± {x['F1_macro_Std']:.3f}", axis=1)
summary['F1_micro_Mean±Std'] = summary.apply(lambda x: f"{x['F1_micro_Mean']:.3f} ± {x['F1_micro_Std']:.3f}", axis=1)

# === Print Tables ===

print("\n=== Accuracy (mean ± std) ===")
print(summary.pivot(index='Dataset', columns='Method', values='Acc_Mean±Std'))

print("\n=== F1-macro (mean ± std) ===")
print(summary.pivot(index='Dataset', columns='Method', values='F1_macro_Mean±Std'))

print("\n=== F1-micro (mean ± std) ===")
print(summary.pivot(index='Dataset', columns='Method', values='F1_micro_Mean±Std'))

# Fold-level results
print("\n=== F1-micro per Fold ===")
print(df.pivot_table(index=['Dataset', 'Fold'], columns='Method', values='F1_micro').round(3))

print("\n=== Accuracy per Fold ===")
print(df.pivot_table(index=['Dataset', 'Fold'], columns='Method', values='Accuracy').round(3))

print("\n=== F1-macro per Fold ===")
print(df.pivot_table(index=['Dataset', 'Fold'], columns='Method', values='F1_macro').round(3))

# Optional export:
# summary.to_csv("benchmark_5fold_results.csv", index=False)
