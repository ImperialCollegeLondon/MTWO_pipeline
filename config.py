import os


# ------------------Hyperparameters-------------------------
WINDOW_SIZE = 5  # seconds
OVERLAPPING_PERCENTAGE = 0.5  # 50% overlapping
SEED = 42 # Random seed for reproducibility
LAB_SAMPLING_RATE = 1500  # Hz
AW_SAMPLING_RATE = 20  # Hz, Axivity sampling rate



# ------------------Directories-----------------------------
# Root directory for the project
# All the following directories are sub-directories under this root directory
rootDir = '/Users/yufeng/Library/CloudStorage/OneDrive-ImperialCollegeLondon/IC/70007 Individual Project'


# Directory for cache files used to fasten data loading
# cache_dir = os.path.join(rootDir, r"MTWO/cache")
cache_dir = os.path.join(rootDir, r"MTWO/old_cache")
# Directory of scaler, label encoder and PCA model
encode_path = os.path.join(cache_dir, 'label_encoder.pkl')
scaler_path = os.path.join(cache_dir, "scaler.joblib")
pca_model_path = os.path.join(cache_dir, 'pca_model.joblib')


# Directory of your training data
movement_dir = os.path.join(rootDir, r'Data/Movement data')
transport_dir = os.path.join(rootDir, r'Data/Transport data') # TODO
walking_dir= os.path.join(rootDir, r'Data/Walking data') # TODO
others_dir = os.path.join(rootDir, r'Data/Others data')


# Directory of the training data from original data sources
# Directory of the Axiviity data
ax_data_dir = os.path.join(rootDir, r"Data/ontrack-activity-classifier/training_data")
# Directory of the lab data
lab_data_dir = os.path.join(rootDir, r"Data/OnTrack")
# Positions of new Transport data
ax_newT_xsl = os.path.join(rootDir, r"Data/MTWO_transport_0424/data_transport_index_0424.xlsx")
ax_newT_csv = os.path.join(rootDir, r"Data/MTWO_transport_0424/data_transport_index_0424.csv")


# Directory to save the trained models
models_dir = os.path.join(rootDir, r"saved_models/ML")
# Directory to save the training comparison results csv
save_dir = os.path.join(rootDir, r'saved_models/ML/training results')





# ------------------Models---------------------------------
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
models = {
    'mlp': MLPClassifier(
        hidden_layer_sizes = (128,),
        max_iter = 800
    ),
    'lr': LogisticRegression(max_iter=500),
    'rf': RandomForestClassifier(
        n_estimators=200,
        criterion='log_loss',
        max_depth=6,
        min_samples_split=2,  # Minimum number of samples required to split an internal node
        min_samples_leaf=1, # Minimum number of samples required to be at a leaf node
        max_leaf_nodes=None, # Maximum number of leaf nodes
        min_weight_fraction_leaf=0.0, 
        max_samples=None, # If not None, draw max_samples from X to train each base estimator.
        max_features='sqrt',
        ),
    'svm': SVC(probability=True, random_state=SEED),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'xgboost' : xgb.XGBClassifier(
            objective='multi:softprob', # multi-class classification, softmax output
            num_class=4, # Number of classes
            learning_rate=0.05,
            max_depth=6, 
            n_estimators=300, # Number of trees
            subsample=0.8, # Fraction of samples to use for each tree
            colsample_bytree=0.8, # Fraction of features to use for each tree
            eval_metric='mlogloss', # Evaluation metric, mlogloss=multi-class cross-entropy
        ),
    # xgboost2: 2-class classification for MO
    'xgboost2' : xgb.XGBClassifier(
        objective='binary:logistic', # 2-class classification
        learning_rate=0.05,
        max_depth=6, 
        n_estimators=300, # Number of trees
        subsample=0.8, # Fraction of samples to use for each tree
        colsample_bytree=0.8, # Fraction of features to use for each tree
        eval_metric='logloss', # Evaluation metric, logloss=binary log loss
        )
    }