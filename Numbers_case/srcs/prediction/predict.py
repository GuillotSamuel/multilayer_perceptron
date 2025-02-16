import os
import sys
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_g import MODEL_PATH, MODEL_FILE, MODEL_PATH, MODEL_FILE, PREDICTION_PATH
from srcs.training.activation import Activation
from srcs.training.utils import Utils

class Predicting:

    def __init__(self, image_dir) -> None:
        """Initialize Predicting class and launch predict function."""
        self.parameters, self.config = Utils.load_model(MODEL_PATH, MODEL_FILE)
        self.layers = self.config['layers']
        self.activation = self.config['activation']

        self.normalization_min = self.config['normalization_min']
        self.normalization_max = self.config['normalization_max']
        
        self.image_paths = [os.path.join(image_dir, f) 
                          for f in os.listdir(image_dir) if f.endswith('.png')]
                
        self.predictions = self.process_and_predict()
        
        self.print_predictions()


    def process_image(self, image_path):
        """Process image for prediction (PNG to normalized pixels)."""
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((28, 28))
            
            img_array = np.array(img).flatten().astype(np.float32)
            
            img_normalized = (img_array - self.normalization_min) / (self.normalization_max - self.normalization_min + 1e-8)

            return img_normalized
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    
    def process_and_predict(self):
        """Process images and predict the class."""
        predictions = []
        for img_path in self.image_paths:
            processed_img = self.process_image(img_path)
            
            if processed_img is not None:
                processed_img_array = np.array(processed_img)
                probabilities, _ = self.forward_propagation(processed_img_array.reshape(1, -1))
                pred_class = np.argmax(probabilities)
                confidence = np.max(probabilities)
                
                predictions.append({
                    'File': os.path.basename(img_path),
                    'Prediction': pred_class,
                    'Trust': f"{confidence * 100:.2f}%",
                    'Probability': probabilities
                })
                
        return predictions
    
    
    def forward_propagation(self, X):
        """Forward propagation for prediction."""
        cache = {'A0': X.copy()}
        A = X

        for i in range(len(self.layers) - 1):
            layer_idx = i + 1
            W = self.parameters[f"W{layer_idx}"]
            b = self.parameters[f"b{layer_idx}"]

            Z = np.dot(A, W.T) + b.T
            A = Activation.activation_g(Z, self.activation[i], derivative=False)
            cache[f"A{layer_idx}"] = A.copy()

        return A, cache
    
    
    def print_predictions(self):
        """Display the predictions."""
        print("Predictions results:")
        print("-" * 50)
        for result in self.predictions:
            print(f"Image: {result['File']}")
            print(f"Prediction: {result['Prediction']}")
            print(f"Trust: {result['Trust']}")
            print("-" * 50)
    

if __name__ == "__main__":
    if not os.path.exists(PREDICTION_PATH):
        os.makedirs(PREDICTION_PATH)

    print(f"Loading files from {PREDICTION_PATH}")
    if os.path.exists(PREDICTION_PATH) and os.listdir(PREDICTION_PATH):
        Predicting(PREDICTION_PATH)
    else:
        raise FileNotFoundError(f"No files found in {PREDICTION_PATH}")
        