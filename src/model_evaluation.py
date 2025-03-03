# src/model_evaluation.py
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test, target_names):
    """Evaluates the trained model."""
    y_test_encoded = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=target_names)
    print(f"Accuracy: {accuracy}")
    print(report)
    return accuracy, report