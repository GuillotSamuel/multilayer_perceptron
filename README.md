# Multilayer perceptron

## Description 📌

This project is developed as part of the 42 school curriculum and implements a multilayer perceptron (MLP) for various classification tasks, including digit recognition and biomedical data analysis.

## Installation ⚙️

1. **Clone the repository**
```bash
git clone https://github.com/GuillotSamuel/multilayer_perceptron
```

2. **Install dependencies**
```bash
make configurations
```

3. **Download the database for number recognition**

For the number recognition case, download the digit image dataset from the following link: [Numérical Images Dataset - Kaggle](https://www.kaggle.com/datasets/pintowar/numerical-images)

## Usage 🚀

Run the projects using in their root folders (Numbers_case and WDBC_case):
```bash
make
```

## Project presentation 📊

### Key notions 🧠

1. **Multilayer perceptron** - A deep learning model composed of multiple layers of neurons, enabling the network to learn complex patterns and make accurate predictions.

![Screenshot of a simple multilayer perceptron with a single output](/documentation/mlp.png)

2. **Perceptron** - The simplest type of artificial neural network, consisting of a single layer that performs linear classification.

3. **Feedforward** - The process of passing input data through the network layers to generate predictions.

4. **Backpropagation** - A learning algorithm that adjusts the network's weights by calculating errors and propagating them backward.

5. **Gradient descents** - An optimization algorithm that minimizes the error function by updating weights in the direction of the steepest descent.

### Project structure 📂

1. **Data processing** - Loading a csv (WDBC case) or multiple png (Number recognition case).

2. **Model** - Using a neuronal network with backpropagation and gradient descents.

4. **Prediction making** - In both cases, you can put examples in prediction folder.

### Bonus Features 🎁

1. **Early Stopping** - Prevents overfitting by stopping training when validation loss stops improving.

2. **Digit Recognition**
    - Imports CSV files and transforms them into pixel arrays.
    - Uses categorical crossentropy + softmax + ReLU.
    - Predicts handwritten digit values.

3. **Dropout** - A regularization technique to reduce overfitting by randomly disabling neurons during training.

4. **Loss Methods**
    - Categorical Crossentropy - Commonly used for multi-class classification tasks.
    - One additional method for alternative optimization.

5. **Weight Initialization**
    - HeUniform - A method designed for deep networks with ReLU activations.
    - Four other initialization methods to compare performance.

6. **Activation Functions**
    - Softmax - Converts logits into probability distributions for classification.
    - Sigmoid - Squashes values between 0 and 1, useful for binary classification.
    - Eight other activation functions for diverse applications.
