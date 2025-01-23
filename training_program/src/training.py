import os
import sys
import pandas as pd
import numpy as np
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import RAW_DATA_PATH, RAW_DATA_FILE, PROCESSED_DATA_PATH, TRAINING_DATA_FILE, VALIDATION_DATA_FILE, LOGS_FOLDER, LOSS_LOGS_FILE, LAYER, EPOCHS, LOSS, BATCH_SIZE, LEARNING_RATE


class TrainingManager:
    
    def __init__(self) -> None:
        """
        Constructor: Sets up the TrainingManager.
        """
        args = self.parse_arguments()
        
        self.layers = args.layer
        self.activation = args.activation
        self.weight_initializer = args.weight_initializer
        self.epochs = args.epochs
        self.loss_function = args.loss
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        
    
    def parse_arguments(self) -> argparse.Namespace:
        """
        Parses command-line arguments for training configuration.

        Returns:
            argparse.Namespace: Parsed arguments including:
                - `--layer` (list[int]): Sizes of each layer (default: LAYERS).
                - `--epochs` (int): Number of training epochs (default: EPOCHS).
                - `--loss` (str): Loss function (choices: "binaryCrossentropy", "categoricalCrossentropy", "meanSquaredError"; default: LOSS).
                - `--batch_size` (int): Batch size (default: BATCH_SIZE).
                - `--learning_rate` (float): Learning rate (default: LEARNING_RATE).

        Example:
            python train.py --config_file config.txt
            python train.py --layer 32 64 32 --epochs 50 --loss binaryCrossentropy --batch_size 16 --learning_rate 0.01
        """
        parser = argparse.ArgumentParser(prog="Multilayer perceptron model training program",
                                         description="Training Manager Configuration",
                                         epilog="")
        parser.add_argument(
            "--layer",
            type=int,
            nargs="+",
            help="List of integers specifying the size of each layer, e.g., --layer 24 24 24.",
            default=LAYER
        )
        parser.add_argument(
            "--epochs", 
            type=int, 
            help="Number of epochs for training, e.g., --epochs 84.",
            default=EPOCHS
        )
        parser.add_argument(
            "--loss", 
            type=str, 
            choices=["binaryCrossentropy", "categoricalCrossentropy", "meanSquaredError"], 
            help="Loss function to use, e.g., --loss binaryCrossentropy. (choices available: \"binaryCrossentropy\", \"categoricalCrossentropy\", \"meanSquaredError\")",
            default=LOSS
        )
        parser.add_argument(
            "--batch_size", 
            type=int, 
            help="Batch size for training, e.g., --batch_size 8.",
            default=BATCH_SIZE
        )
        parser.add_argument(
            "--learning_rate", 
            type=float, 
            help="Learning rate for the optimizer, e.g., --learning_rate 0.0314.",
            default=LEARNING_RATE
        )        
        parser.add_argument(
            "--config_file", 
            type=str, 
            help="Path to a configuration file that contains the training settings."
        )
        args = parser.parse_args()

        if len(args.layer) > 1:
            args.activation = ["sigmoid"] * (len(args.layer) - 1) + ["softmax"]
            args.weight_initializer = ["heUniform"] * (len(args.layer) - 1)
        else:
            args.activation = ["sigmoid"]

        if args.config_file:
            if len(sys.argv) == 2:
                return self.parse_config_file(args)
            else:
                raise ValueError("Can't have a config_file and other arguments at the same time.")
        else:
            return args


    def parse_config_file(args: argparse.Namespace) -> argparse.Namespace:
        """
        Parses the configuration file and returns the parsed arguments.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            argparse.Namespace: Parsed arguments from the config file.
        """
        pass