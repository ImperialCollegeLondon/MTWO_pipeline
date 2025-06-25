import joblib
import os
import numpy as np
import sys
from sklearn.decomposition import PCA
from loguru import logger

# Add the parent directory to the Python path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cache_dir, SEED, pca_model_path

# Configure loguru logger
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

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
    logger.info(f"Number of features after PCA: {X_train_pca.shape[1]}")
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
    logger.info(f"PCA model saved to {pca_model_path}.")

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

    logger.info(f"Explained variance ratio: {pca_model.explained_variance_ratio_}")
    logger.info(f"Total explained variance: {np.sum(pca_model.explained_variance_ratio_)}")

    return X_train_scaled, X_test_scaled