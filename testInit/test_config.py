import os

from config import models_dir,rootDir

root_save_dir = os.path.join(rootDir,'MTWO_pipeline/pred_res')

xgboost2_path = os.path.join(models_dir, 'xgboost2.pkl')  # Assuming xgboost2 is used for MO mode
xgboost_path = os.path.join(models_dir, 'xgboost.pkl')
rf_path = os.path.join(models_dir, 'rf.pkl')
mlp_path = os.path.join(models_dir, 'mlp.pkl')
lr_path = os.path.join(models_dir, 'lr.pkl')
svm_path = os.path.join(models_dir, 'svm.pkl')
knn_path = os.path.join(models_dir, 'knn.pkl')

model_dics = {'xgboost': xgboost_path, 'rf': rf_path, 'mlp': mlp_path, 'lr': lr_path, 'svm': svm_path, 'knn': knn_path}

ground_truth_dic = {
            '1': "M",
            '2': "W",
            '3': "O",
            '6': "T",
            # 4: "CHECK"
        }

model_dics_mo = {
    'xgboost2': xgboost2_path, 
    'rf': rf_path, 
    'mlp': mlp_path, 
    'lr': lr_path, 
    'svm': svm_path, 
    'knn': knn_path
}

ground_truth_dic_mo = {
    '1': "M",
    '2': "O",
}