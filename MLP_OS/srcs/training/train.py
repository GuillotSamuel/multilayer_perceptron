import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, TRAINING_RESULTS_FILE, VALIDATION_RESULTS_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE, MODEL_PATH, MODEL_FILE

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
        # Utils.save_model()


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
        X = Data_preprocessor.normalize_data(self.training_data)
        Y = Data_preprocessor.one_hot_encode(self.training_results)
        X_val = Data_preprocessor.normalize_data(self.validation_data)
        Y_val = Data_preprocessor.one_hot_encode(self.validation_results)
        Initialization.initialize_weights(self.parameters, self.layers, self.weight_initializer)

        for epoch in range(self.epochs):
            A, cache = self.forward_propagation(self.parameters, X)
            
            loss = Cost.compute_loss(A, Y, self.loss)
            self.losses_train.append(loss)

            accuracy = self.compute_accuracy(A, Y)
            self.accuracies_train.append(accuracy)

            gradients = self.back_propagation(X, Y, self.parameters, cache)
            self.update_weights(self.parameters, gradients)
            
            self.validate_training(X_val, Y_val, epoch)
            
            if epoch % 1000 == 0 or epoch == self.epochs - 1:
                self.print_predictions_comparison(A, Y, num_samples=30)


    def validate_training(self, X_val, Y_val, epoch) -> None:
        """  """
        A_val, _ = self.forward_propagation(self.parameters, X_val)
        val_loss = Cost.compute_loss(A_val, Y_val, self.loss)
        val_accuracy = self.compute_accuracy(A_val, Y_val)
        
        self.losses_validation.append(val_loss)
        self.accuracies_validation.append(val_accuracy)
        
        if epoch % 1000 == 0 or epoch == self.epochs - 1:
            print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy*100:.2f}%")
        

    def create_logs(self) -> None:
        # Plotting the loss
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.losses_train, label='Training Loss', color='blue')
        plt.plot(self.losses_validation, label='Validation Loss', color='orange')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plotting the accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies_train, label='Training Accuracy', color='blue')
        plt.plot(self.accuracies_validation, label='Validation Accuracy', color='orange')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_logs.png')  # Save the plot as an image file
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
        # Convertir les one-hot en labels de classe
        Y_pred_labels = np.argmax(A, axis=1)
        Y_true_labels = np.argmax(Y, axis=1)

        # Calculer l'accuracy globale
        accuracy = np.mean(Y_pred_labels == Y_true_labels)

        # Sélectionner un sous-échantillon
        indices = np.random.choice(len(Y), num_samples, replace=False)

        # Créer un DataFrame pour l'affichage
        results = pd.DataFrame({
            'Sample Index': indices,
            'Predicted Class': Y_pred_labels[indices],
            'Actual Class': Y_true_labels[indices],
            'Correct': Y_pred_labels[indices] == Y_true_labels[indices]
        })

        # Ajouter les probabilités de chaque classe
        prob_df = pd.DataFrame(A[indices], columns=[f'Class {i}_prob' for i in range(A.shape[1])])
        results = pd.concat([results, prob_df], axis=1)

        print(f"\n=== Predictions vs Actual Values (Sample) - Global Accuracy: {accuracy*100:.2f}% ===")
        print(results.to_string(index=False))
        print("===========================================\n")


if __name__ == "__main__":
    Training()
