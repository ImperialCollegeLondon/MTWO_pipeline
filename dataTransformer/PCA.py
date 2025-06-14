import joblib
import os
import numpy as np
from sklearn.decomposition import PCA

from config import cache_dir, SEED, pca_model_path

def apply_pca(X_train, X_test, n_components=0.95, plot_variance=False):
    """
    Reducing the dimensionality of the training and test sets using PCA.
    n_components could be 
        an integer (number of components) or 
        a float between 0 and 1 (explained variance ratio).
    """
    pca = PCA(n_components=n_components, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"[info@PCA.apply_pca] -> Number of features after PCA: {X_train_pca.shape[1]}")
    if plot_variance:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of principal components') # 主成分数
        plt.ylabel('Cumulative explained variance ratio') # 累计解释方差比
        plt.title('PCA cumulative explained variance ratio') # PCA累计解释方差比
        plt.grid(True)
        plt.show()
    return X_train_pca, X_test_pca, pca

def pca(X_train_scaled, X_test_scaled, n_components=0.95, vis=False):
    X_train_scaled, X_test_scaled, pca_model = apply_pca(X_train_scaled, X_test_scaled, n_components=n_components, plot_variance=vis)
    # Save the PCA model
    joblib.dump(pca_model, pca_model_path, compress=3)
    print(f"[info@PCA.pca] -> PCA model saved to {pca_model_path}.")

    if vis:
        # Draw heatmap of PCA components
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        sns.heatmap(np.abs(pca_model.components_), cmap='viridis', annot=False)
        plt.xlabel('Original index of features')
        plt.ylabel('Index of principal components')
        plt.title('Heatmap of PCA components')
        plt.show()

    print(f"[info@PCA.pca] -> Explained variance ratio: {pca_model.explained_variance_ratio_}")
    print(f"[info@PCA.pca] -> Total explained variance: {np.sum(pca_model.explained_variance_ratio_)}")

    return X_train_scaled, X_test_scaled, pca_model