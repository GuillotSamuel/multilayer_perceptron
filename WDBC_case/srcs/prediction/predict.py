import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import MODEL_PATH, MODEL_FILE, PREDICTION_PATH, PREDICTION_FILE
from srcs.training.activation import Activation
from srcs.training.data_preprocessor import Data_preprocessor
from srcs.training.utils import Utils

class Predicting:

    def __init__(self, new_data) -> None:
        """Initialize Predicting class and launch predict function."""
        self.parameters, self.config = Utils.load_model(MODEL_PATH, MODEL_FILE)
        self.layers = self.config['layers']
        self.activation = self.config['activation']
        self.normalization_mean = self.config['normalization_mean']
        self.normalization_std = self.config['normalization_std']
        self.normalization_min = self.config['normalization_min']
        self.normalization_max = self.config['normalization_max']

        self.new_data = new_data

        self.has_labels = 'diagnosis' in self.new_data.columns
        
        if self.has_labels:
            self.true_labels = self.new_data['diagnosis'].map({'B': 0, 'M': 1}).values
        
        self.detailed_results = self.predict_and_explain(self.new_data, num_samples=10)

        print("Prediction Results:")
        print(self.detailed_results)

        if self.has_labels:
            bce_error = self.evaluate_binary_cross_entropy(self.detailed_results)
            print(f"\nBinary Cross-Entropy Error: {bce_error:.6f}")


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict if a patient has a cancer."""
        X_copy = X.copy()
        
        if 'diagnosis' in X_copy.columns:
            X_features = X_copy.drop(columns=['diagnosis'])
        else:
            X_features = X_copy
        
        X_processed = Data_preprocessor.normalize_data(X_features,
                                                       self.normalization_min, self.normalization_max,
                                                       self.normalization_mean, self.normalization_std,
                                                       method='minmax')
        
        probabilities, _ = self.forward_propagation(X_processed)
        predictions = np.argmax(probabilities, axis=1)

        return predictions, probabilities


    def forward_propagation(self, X):
        """Forward propagation to obtain the predictions."""
        cache = {'A0': X.copy()}
        A = X

        for i in range(len(self.layers) - 1):
            layer_idx = i + 1
            W = self.parameters[f"W{layer_idx}"]
            b = self.parameters[f"b{layer_idx}"]

            Z = np.dot(A, W.T) + b.T
            cache[f"Z{layer_idx}"] = Z

            A = Activation.activation_g(Z, self.activation[i], derivative=False)
            cache[f"A{layer_idx}"] = A.copy()

        return A, cache


    def predict_and_explain(self, X, num_samples=None):
        """Predict and explain the predictions."""
        predictions, probabilities = self.predict(X)

        results = pd.DataFrame()

        for i in range(probabilities.shape[1]):
            results[f'Probability_Class_{i}'] = probabilities[:, i]

        results['Predicted_Class'] = predictions
        results['Confidence'] = np.max(probabilities, axis=1)
        
        results['Diagnostic'] = results['Predicted_Class'].map({
            0: 'B',
            1: 'M'
        })
        
        if self.has_labels:
            results['True_Class'] = self.true_labels
            results['True_Diagnostic'] = X['diagnosis']
            results['Correct'] = results['Predicted_Class'] == results['True_Class']

        return results

    def evaluate_binary_cross_entropy(self, results):
        """Calculate binary cross-entropy error."""
        if 'True_Class' not in results.columns:
            print("Warning: Cannot calculate binary cross-entropy without true labels")
            return None
        
        y_true = results['True_Class'].values
        p_pred = results['Probability_Class_1'].values
        
        epsilon = 1e-15
        p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
        
        bce_per_sample = -(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))
        
        bce = np.mean(bce_per_sample)
        
        return bce


if __name__ == "__main__":
    prediction_file_path = os.path.join(PREDICTION_PATH, PREDICTION_FILE)
    print(f"Loading data from: {prediction_file_path}")
    if os.path.exists(prediction_file_path) and os.path.getsize(prediction_file_path) > 0:
        Predicting(pd.read_csv(prediction_file_path))
    else:
        raise FileNotFoundError(f"The file {prediction_file_path} is either missing or empty.")