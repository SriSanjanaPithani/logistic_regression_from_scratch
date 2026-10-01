# Logistic Regression from Scratch

A NumPy implementation of binary and multiclass logistic regression for
handwritten digit classification.

This project explores the mathematical foundations of logistic regression
by implementing the models and optimization algorithms from scratch rather
than relying on machine learning libraries for training.

## Features

- Binary logistic regression using the sigmoid function
- Multiclass logistic regression using softmax
- Batch Gradient Descent (BGD)
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent
- Cross-entropy gradient computation
- Hyperparameter tuning using validation accuracy
- Hand-engineered symmetry and intensity image features
- Visualization of learned decision boundaries

## Feature Engineering

Each 16×16 digit image is represented using two features:

**Symmetry** — measures the difference between an image and its
horizontally flipped version.

**Intensity** — measures the total pixel intensity of the image.

A bias term is also included for the logistic regression models.

## Binary Classification

The binary classifier distinguishes handwritten digits 1 and 2 using
sigmoid logistic regression.

![Binary Decision Boundary](figures/binary_decision_boundary.png)

The model learns a linear decision boundary based on the symmetry and
intensity features.

### Binary Results

The model was evaluated using learning rates of 0.05, 0.25, and 0.75
and maximum iteration values of 50, 250, and 750. Mini-batch gradient
descent with a batch size of 10 was used during hyperparameter tuning.

The best configuration achieved:

- Validation accuracy: **97.88%**
- Test accuracy: **93.07%**
- Learning rate: **0.25**
- Maximum iterations: **750**
- Batch size: **10**

The selected model was evaluated on the test set without retraining.

## Multiclass Classification

Softmax logistic regression extends the model to classify digits 0, 1,
and 2.

![Multiclass Decision Boundaries](figures/multiclass_decision_boundaries.png)

Pairwise decision boundaries illustrate how the learned linear models
separate the three digit classes.

### Multiclass Results

The same learning-rate and iteration search was performed using
mini-batch gradient descent with a batch size of 10.

The best configuration achieved:

- Validation accuracy: **87.78%**
- Test accuracy: **86.85%**
- Learning rate: **0.25**
- Maximum iterations: **250**
- Batch size: **10**

## Optimization Methods

Three gradient-based optimization approaches were implemented and
compared:

- **Batch Gradient Descent (BGD):** updates the model weights once per
  epoch using the average gradient across the training data.
- **Mini-Batch Gradient Descent:** divides the training data into batches
  and updates the weights after each batch.
- **Stochastic Gradient Descent (SGD):** updates the weights after each
  individual training sample.

These implementations provide a comparison of different approaches to
optimizing the same logistic regression models.

## Sigmoid vs. Softmax

The project also compares sigmoid and softmax logistic regression when
classifying only two classes.

When trained on the same binary dataset, both approaches achieved
approximately:

- Training accuracy: **97.19%**
- Validation accuracy: **97.88%**

This demonstrates that softmax logistic regression with two classes
produces equivalent classification behavior to binary sigmoid logistic
regression.

## Training Features

![Training Features](figures/train_features.png)

The visualization shows the distribution of digits 1 and 2 in the
two-dimensional feature space.

## Hyperparameter Tuning

Models are evaluated across different learning rates and training
iterations. Validation accuracy is used to select the best-performing
configuration before evaluating it on the test set.

The hyperparameter search evaluates:

- Learning rates: **0.05, 0.25, 0.75**
- Maximum iterations: **50, 250, 750**
- Mini-batch size: **10**

The best validation model is selected and then evaluated on the unseen
test data without retraining.

## Technologies

- Python
- NumPy
- Matplotlib

## Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify the project structure

Make sure the repository contains:

```text
project/
├── data/
│   ├── training.npz
│   └── test.npz
├── code/
│   ├── main.py
│   ├── DataReader.py
│   ├── LogisticRegression.py
│   └── SoftmaxRegression.py
├── figures/
├── requirements.txt
└── README.md
```

### 6. Navigate to the code directory

```bash
cd code
```

### 7. Run the project

```bash
python main.py
```

The program loads the training and testing datasets, preprocesses the
16×16 digit images, extracts the symmetry and intensity features, trains
the binary and multiclass logistic regression models, performs
hyperparameter tuning, evaluates the selected models, and generates the
project visualizations.