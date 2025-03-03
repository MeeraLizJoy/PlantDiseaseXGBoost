# src/main.py
import data_loading, feature_extraction, model_training, model_evaluation, utils
import os
import numpy as np
from sklearn.decomposition import PCA
from memory_profiler import profile # Import the profile decorator

@profile
def main():
    data_dir = "data/PlantifyDr"
    img_width, img_height = 224, 224
    efficientnet_version = 'B0'
    pca_components = 64

    # Data loading and preprocessing
    train_generator, validation_generator, class_indices = data_loading.create_data_generators(data_dir, img_width, img_height)
    label_encoder = data_loading.preprocess_labels(class_indices)

    # Feature extraction
    feature_extractor = feature_extraction.create_feature_extractor(img_width, img_height, efficientnet_version)

    # Training feature extraction and caching
    X_train_features, y_train = [], []
    train_cache_file = f"data/train_features_{efficientnet_version}.npy"
    for batch_features, batch_y in data_loading.batch_generator(train_generator, feature_extractor, train_cache_file):
        X_train_features.extend(batch_features)
        y_train.extend(batch_y)
    X_train_features = np.array(X_train_features)
    y_train = np.array(y_train)

    # Validation feature extraction and caching
    X_val_features, y_val = [], []
    val_cache_file = f"data/val_features_{efficientnet_version}.npy"
    for batch_features, batch_y in data_loading.batch_generator(validation_generator, feature_extractor, val_cache_file):
        X_val_features.extend(batch_features)
        y_val.extend(batch_y)
    X_val_features = np.array(X_val_features)
    y_val = np.array(y_val)

    # PCA if enabled
    if pca_components:
        pca = PCA(n_components=pca_components)
        X_train_features = pca.fit_transform(X_train_features)
        X_val_features = pca.transform(X_val_features)
        print(f"PCA applied, feature shape reduced to: {X_train_features.shape[1]}")

    print(f"Shape of X_train_features: {X_train_features.shape}")
    print(f"Shape of X_val_features: {X_val_features.shape}")

    # Model training (with hyperparameter tuning)
    xgb_model = model_training.train_xgboost_dask_model(X_train_features, y_train, len(class_indices), tune_hyperparameters=False)

    model_training.save_model(xgb_model, "data/xgb_model.pkl")

    # Model evaluation
    model_evaluation.evaluate_model(xgb_model, X_val_features, y_val, label_encoder.classes_)

if __name__ == "__main__":
    main()