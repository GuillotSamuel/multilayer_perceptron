import numpy as pd

from srcs.training.activation import Activation
from srcs.training.cost import Cost
from srcs.training.data_preprocessor import Data_preprocessor
from srcs.training.initialization import Initialization
from srcs.training.monitoring import Monitoring
from srcs.training.optimisation import Optimisation
from srcs.training.utils import Utils

class Training:

    def __init__(self):
        
        args = Initialization.parse_arguments()

        # MLP Parameters
        self.layers = args.layer
        self.activation = args.activation
        self.weight_initializer = args.weight_initializer
        self.epochs = args.epochs
        self.loss_function = args.loss
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_outputs = Initialization.count_outputs()

        # Logs
        self.losses_train = []
        self.accuracies_train = []
        self.losses_validation = []
        self.accuracies_validation = []
        
        # Training Process
        self.train()
        self.validate_training()
        self.create_logs()
        Utils.save_model()
        
        
    def train(self) -> None:
        pass
    
    
    def validate_training(self) -> None:
        pass
    
    
    def create_logs(self) -> None:
        pass


if __name__ == "__main__":
    Training()