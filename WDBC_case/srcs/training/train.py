import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, LOGS_FOLDER, LOSS_LOGS_FILE, MODEL_PATH, MODEL_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_FILE, EARLY_STOPPING_LIMIT

from srcs.training.activation import Activation
from srcs.training.cost import Cost
from srcs.training.data_preprocessor import Data_preprocessor
from srcs.training.initialization import Initialization
from srcs.training.utils import Utils

class Training:

    def __init__(self) -> None:
        """  """
        
        # MLP Parameters
        self.training_data = Utils.load_file(f"{PROCESSED_DATA_PATH}/{TRAINING_DATA_FILE}")
        self.training_results = Utils.load_file(f"{PROCESSED_DATA_PATH}/{TRAINING_RESULTS_FILE}")
        self.validation_data = Utils.load_file(f"{PROCESSED_DATA_PATH}/{VALIDATION_DATA_FILE}")
        self.validation_results = Utils.load_file(f"{PROCESSED_DATA_PATH}/{VALIDATION_RESULTS_FILE}")
        
        args = Initialization.parse_arguments(self.training_data, self.validation_data,
                                              self.training_results, self.validation_results)

        self.layers = args.layer
        self.epochs = args.epochs
        self.loss = args.loss
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.activation = args.activation
        self.weight_initializer = args.weight_initializer
        self.num_inputs = args.num_inputs
        self.num_outputs = args.num_outputs
        
        self.print_variables() # TEST
        
        self.parameters = {}

        # Logs
        self.losses_train = []
        self.accuracies_train = []
        self.losses_validation = []
        self.accuracies_validation = []
        
        # Training Process
        self.train()
        self.create_logs()
        
        # Saving the Model
        self.model_config = {
            'layers': self.layers,
            'epochs': self.epochs,
            'loss': self.loss,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'weight_initializer': self.weight_initializer,
            'num_inputs': self.num_inputs,
            'num_outputs': self.num_outputs,
            'normalization_mean': self.training_data.mean(),
            'normalization_std': self.training_data.std(),
            'normalization_min': self.training_data.min(),
            'normalization_max': self.training_data.max()
        }        
        Utils.save_model(MODEL_PATH, MODEL_FILE, self.parameters, self.model_config)


    def print_variables(self) -> None: # TEST
        """
        Print all variables of the object in a readable format.
        """
        print("\n--- Training Manager Variables ---\n")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
        print("-----------------------------------")


    def train(self) -> None:
        """  """
        X = Data_preprocessor.normalize_data(self.training_data,
                                             self.training_data.min(), self.training_data.max(),
                                             self.training_data.mean(), self.training_data.std(),
                                             method='minmax')
        Y = Data_preprocessor.one_hot_encode(self.training_results)
        X_val = Data_preprocessor.normalize_data(self.validation_data,
                                                 self.training_data.min(), self.training_data.max(),
                                                 self.training_data.mean(), self.training_data.std(),
                                                 method='minmax')
        Y_val = Data_preprocessor.one_hot_encode(self.validation_results)
        Initialization.initialize_weights(self.parameters, self.layers, self.weight_initializer)
        
        X_len = X.shape[0]

        X = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        Y = Y.to_numpy() if hasattr(Y, 'to_numpy') else np.array(Y)
      
        for epoch in range(self.epochs):
            permutation = np.random.permutation(X_len)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            
            for i in range(0, X_len, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                Y_batch = Y_shuffled[i:i+self.batch_size]
            
                A_batch, cache_batch = self.forward_propagation(self.parameters, X_batch)

                gradients = self.back_propagation(X_batch, Y_batch, self.parameters, cache_batch)

                self.update_weights(self.parameters, gradients)

            A_train, _ = self.forward_propagation(self.parameters, X)
            loss_train = Cost.compute_loss(A_train, Y, self.loss)
            accuracy_train = self.compute_accuracy(A_train, Y)
            self.losses_train.append(loss_train)
            self.accuracies_train.append(accuracy_train)

            if self.validate_training(X_val, Y_val, epoch) == True:
                break

            if epoch % 200 == 0 or epoch == self.epochs - 1:
                self.print_predictions_comparison(A_train, Y, num_samples=30)


    def validate_training(self, X_val, Y_val, epoch) -> bool:
        """ Use model weights and bias to validate the training """
        A_val, _ = self.forward_propagation(self.parameters, X_val)
        val_loss = Cost.compute_loss(A_val, Y_val, self.loss)
        val_accuracy = self.compute_accuracy(A_val, Y_val)
        
        self.losses_validation.append(val_loss)
        self.accuracies_validation.append(val_accuracy)
        
        if self.early_stopping(epoch) == True:
            return True

        if epoch % 200 == 0 or epoch == self.epochs - 1:
            print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy*100:.2f}%")
            
        return False
        

    def early_stopping(self, epoch) -> bool:
        """ Check if the model should stop training early """
        if EARLY_STOPPING_LIMIT == 0:
            return False
        if epoch > EARLY_STOPPING_LIMIT:
            last_losses = self.losses_validation[-EARLY_STOPPING_LIMIT:]
            current_loss = last_losses[-1]
            previous_losses = last_losses[:-1]
            
            if all(current_loss > prev_loss for prev_loss in previous_losses):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                return True
        
        return False


    def create_logs(self) -> None:
        """Create Loss and Accuracy graphs."""
        def smooth_curve(points, factor=0.97):
            """Applies exponential smoothing to a list of points."""
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points

        smoothed_losses_train = smooth_curve(self.losses_train)
        smoothed_losses_validation = smooth_curve(self.losses_validation)
        smoothed_accuracies_train = smooth_curve(self.accuracies_train)
        smoothed_accuracies_validation = smooth_curve(self.accuracies_validation)
        
        # Loss
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(smoothed_losses_train, label='Training Loss', color='blue')
        plt.plot(smoothed_losses_validation, label='Validation Loss', color='orange')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(smoothed_accuracies_train, label='Training Accuracy', color='blue')
        plt.plot(smoothed_accuracies_validation, label='Validation Accuracy', color='orange')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(LOGS_FOLDER, LOSS_LOGS_FILE))
        plt.show()


    """ ---------- TRAINING FUNCTIONS ---------- """

    def forward_propagation(self, parameters, X):
        """  """
        cache = {'A0': X.copy()}
        A = X

        for i in range(len(self.layers) - 1):
            layer_idx = i + 1
            W = parameters[f"W{layer_idx}"]
            b = parameters[f"b{layer_idx}"]

            Z = np.dot(A, W.T) + b.T
            cache[f"Z{layer_idx}"] = Z

            A = Activation.activation_g(Z, self.activation[i], derivative = False)

            cache[f"A{layer_idx}"] = A.copy()

        return A, cache
    

    def compute_accuracy(self, A, Y):
        Y_pred_labels = np.argmax(A, axis=1)
        Y_true_labels = np.argmax(Y, axis=1)
        return np.mean(Y_pred_labels == Y_true_labels)


    def back_propagation(self, X, Y, parameters, cache):
        gradients = {}
        m = X.shape[0]
        num_layers = len(self.layers) - 1

        AL = cache[f'A{num_layers}']
        dZ = AL - Y

        A_prev = cache[f'A{num_layers-1}']
        gradients[f'dW{num_layers}'] = (1/m) * np.dot(dZ.T, A_prev)
        gradients[f'db{num_layers}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True).T

        for l in reversed(range(1, num_layers)):
            W_next = parameters[f'W{l+1}']
            dA = np.dot(dZ, W_next)

            Z_current = cache[f'Z{l}']
            activation_type = self.activation[l-1]
            dZ = dA * Activation.activation_g(Z_current, activation_type, derivative=True)

            A_prev = cache[f'A{l-1}']
            gradients[f'dW{l}'] = (1/m) * np.dot(dZ.T, A_prev)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True).T

        return gradients


    def update_weights(self, parameters, gradients):
        L = len(self.layers) - 1
        for l in range(1, L+1):
            parameters[f"W{l}"] -= self.learning_rate * gradients[f"dW{l}"]
            parameters[f"b{l}"] -= self.learning_rate * gradients[f"db{l}"]


    def print_predictions_comparison(self, A, Y, num_samples=10): # TEST
        """
        Affiche un échantillon des prédictions vs les vraies valeurs
        """
        Y_pred_labels = np.argmax(A, axis=1)
        Y_true_labels = np.argmax(Y, axis=1)

        accuracy = np.mean(Y_pred_labels == Y_true_labels)

        indices = np.random.choice(len(Y), num_samples, replace=False)

        results = pd.DataFrame({
            'Sample Index': indices,
            'Predicted Class': Y_pred_labels[indices],
            'Actual Class': Y_true_labels[indices],
            'Correct': Y_pred_labels[indices] == Y_true_labels[indices]
        })

        prob_df = pd.DataFrame(A[indices], columns=[f'Class {i}_prob' for i in range(A.shape[1])])
        results = pd.concat([results, prob_df], axis=1)

        print(f"\n=== Predictions vs Actual Values (Sample) - Global Accuracy: {accuracy*100:.2f}% ===")
        print(results.to_string(index=False))
        print("===========================================\n")


if __name__ == "__main__":
    Training()
