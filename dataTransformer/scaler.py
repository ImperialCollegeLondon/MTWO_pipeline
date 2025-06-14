from sklearn.preprocessing import StandardScaler
import joblib
import os

from config import cache_dir, scaler_path

def scale(X_train, X_test, return_scaler=False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    print(f"[info@scaler] -> Scaler saved to {scaler_path}.")

    if return_scaler:
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, X_test_scaled