# src/model_training.py
import xgboost as xgb
import pickle
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client

def train_xgboost_dask_model(X_train, y_train, num_classes, tune_hyperparameters=False):
    """Trains XGBoost model using dask_xgboost."""
    y_train_encoded = np.argmax(y_train, axis=1)

    # Create a Dask DataFrame and Series
    X_dask = dd.from_array(X_train)
    y_dask = dd.from_array(y_train_encoded)

    # Start a Dask Client
    client = Client()

    if tune_hyperparameters:
        # Hyperparameter tuning with dask_xgboost is more complex
        # For simplicity, we'll skip it for now.
        pass
    else:
        xgb_params = {
            'objective': 'multi:softmax',
            'num_class': num_classes,
        }
        # Train the XGBoost model using xgboost.dask
        bst = xgb.dask.train(client, xgb_params, X_dask, y_dask)

        # Extract the trained model
        bst = bst['booster']

        # Close the client
        client.close()

        return bst

def save_model(model, filepath):
    """Saves the trained model."""
    pickle.dump(model, open(filepath, 'wb'))