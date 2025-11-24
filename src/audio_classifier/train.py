from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import joblib
import json
import sys
import os

from .__utils__ import load_data_for_svm, evaluate_svm_model

def train_svm_classifier(data_dir="data/processed", config_path="src/svm_config.json", model_dir="models"):

    os.makedirs(model_dir, exist_ok=True)

    data = load_data_for_svm(data_dir)
    
    X_train_full = np.concatenate((data["train"]["X"], data["validation"]["X"]), axis=0)
    y_train_full = np.concatenate((data["train"]["y"], data["validation"]["y"]), axis=0)
    
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    with open(config_path, 'r') as f:
        param_grid = json.load(f)

    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('svm', SVC(class_weight='balanced', probability=True))
        ])
    
    pipeline_params = {f"svm__{k}": v for k, v in param_grid.items()}
    
    grid_search = GridSearchCV(
        pipeline, 
        pipeline_params, 
        cv=3, 
        n_jobs=12, 
        verbose=2,
        scoring='f1_macro',
        refit='f1_macro'
    )

    grid_search.fit(X_train_full, y_train_full)

    best_model = grid_search.best_estimator_
    evaluate_svm_model(best_model, X_test, y_test)

    save_path = os.path.join(model_dir, "best_svm_model.pkl")
    joblib.dump(best_model, save_path)

def main():
    train_svm_classifier()

if __name__ == "__main__":
    main()