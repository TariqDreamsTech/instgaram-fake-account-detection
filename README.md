# Fake Account Detection - Machine Learning App

This application uses advanced machine learning models to detect fake Instagram accounts based on various account features. Includes both basic ML models and state-of-the-art 2025 techniques with deep learning and hyperparameter optimization.

## Features

### Basic Models
- Train multiple ML models (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- Compare model performance
- Predict if an account is fake or real
- Interactive command-line interface

### Advanced Models (2025 Best Practices)
- **Deep Neural Networks** with TensorFlow/Keras
  - Multi-layer architecture with Batch Normalization
  - Dropout regularization
  - Early stopping and learning rate scheduling
- **Hyperparameter Tuning**
  - RandomizedSearchCV for Random Forest
  - Optuna optimization for XGBoost
  - Cross-validation for robust evaluation
- **Ensemble Methods**
  - Voting Classifier (soft voting)
  - Stacking Classifier with meta-learner
- **Advanced Feature Engineering**
  - Engagement ratio
  - Following-follower ratio
  - Username density features
- **Visualizations**
  - Training history plots for neural networks
  - Feature importance charts
  - Confusion matrices
  - ROC curves

## Installation

First, install all required dependencies:
```bash
pip install -r requirements.txt
```

**Note:** For advanced features (deep learning), ensure you have:
- TensorFlow 2.13+ for neural networks
- XGBoost for gradient boosting
- Optuna for hyperparameter optimization

## Usage

### Option 1: Interactive Menu (Recommended)

Run the main application:
```bash
python main.py
```

This will show a menu with options to:
1. Train Basic Machine Learning Models
2. Train Advanced Models (Deep Learning + Hyperparameter Tuning)
3. Predict if an account is fake (Basic Model)
4. Predict if an account is fake (Advanced Model)
5. Exit

### Option 2: Train Basic Models

Train basic models and save the best one:
```bash
python train_model.py
```

### Option 3: Train Advanced Models

Train advanced models with deep learning and hyperparameter tuning:
```bash
python train_advanced.py
```

This will:
- Train hyperparameter-tuned Random Forest and XGBoost
- Train a deep neural network
- Create ensemble models (Voting and Stacking)
- Generate visualizations (saved in `plots/` directory)
- Save the best performing model

### Option 4: Use Prediction Scripts

Predict with basic model:
```bash
python predict.py
```

Predict with advanced model:
```bash
python predict_advanced.py
```

## Model Training

### Basic Models

The basic training script will:
1. Load the fake account dataset
2. Prepare features and split data into train/test sets
3. Train multiple ML models:
   - Random Forest
   - Gradient Boosting
   - Logistic Regression
   - SVM (Support Vector Machine)
4. Evaluate models using accuracy, precision, recall, and F1-score
5. Save the best performing model

### Advanced Models

The advanced training script includes:

1. **Hyperparameter Optimization**
   - Random Forest: RandomizedSearchCV with 50 iterations
   - XGBoost: Optuna with 30 trials and Bayesian optimization
   - Cross-validation (5-fold) for robust evaluation

2. **Deep Neural Network**
   - Architecture: 128 → 64 → 32 → 16 → 1 neurons
   - Batch Normalization after each dense layer
   - Dropout for regularization (30% dropout rate)
   - Adam optimizer with learning rate scheduling
   - Early stopping to prevent overfitting

3. **Ensemble Methods**
   - **Voting Classifier**: Combines Random Forest, Gradient Boosting, and Logistic Regression
   - **Stacking Classifier**: Uses Random Forest, Gradient Boosting, and SVM as base models with Logistic Regression as meta-learner

4. **Advanced Feature Engineering**
   - Engagement ratio (followers/media count)
   - Following-follower ratio
   - Username density (digits/username length)

5. **Visualizations**
   - Training history plots (accuracy/loss curves)
   - Feature importance charts
   - Confusion matrices
   - ROC curves

The trained models will be saved in the `models/` directory.

## Prediction

To predict if an account is fake, you need to provide the following features:

- `user_media_count`: Number of posts
- `user_follower_count`: Number of followers
- `user_following_count`: Number of accounts following
- `user_has_profil_pic`: Has profile picture (1=Yes, 0=No)
- `user_is_private`: Is private account (1=Yes, 0=No)
- `user_biography_length`: Number of characters in biography
- `username_length`: Number of characters in username
- `username_digit_count`: Number of digits in username

## Example Usage in Code

```python
from predict import predict_from_dict

account_data = {
    "user_media_count": 50,
    "user_follower_count": 150,
    "user_following_count": 2000,
    "user_has_profil_pic": 1,
    "user_is_private": 0,
    "user_biography_length": 50,
    "username_length": 10,
    "username_digit_count": 3,
}

result = predict_from_dict(account_data)
print(f"Is Fake: {result['is_fake']}")
print(f"Fake Probability: {result['fake_probability']:.2f}%")
```

## Model Files

After training, the following files will be created in `models/`:

- `best_model.joblib`: The best performing model
- `scaler.joblib`: Feature scaler (if needed)
- `model_info.joblib`: Model metadata and feature information

## Project Structure

```
ml_app/
├── main.py                  # Main interactive application
├── train_model.py           # Basic model training script
├── train_advanced.py        # Advanced training with deep learning & hyperparameter tuning
├── predict.py               # Basic prediction functions
├── predict_advanced.py      # Advanced prediction with neural network support
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── models/                  # Saved models (created after training)
│   ├── best_model.joblib                    # Basic model (best)
│   ├── model_basic_*.joblib                 # All basic models
│   ├── model_advanced_*.joblib              # All advanced models
│   ├── model_advanced_neural_network.h5     # Neural network model
│   ├── scaler_basic.joblib                  # Basic scaler
│   ├── scaler_advanced.joblib               # Advanced scaler
│   ├── model_info.joblib                    # Basic model info
│   ├── model_info_advanced.joblib           # Advanced model info
│   ├── all_models_metrics_basic.json        # Basic models metrics
│   └── all_models_metrics_advanced.json     # Advanced models metrics
└── plots/                   # Visualization outputs
    ├── neural_network_training.png
    ├── feature_importance.png
    └── ...
```

## Performance Comparison

The advanced models typically achieve:
- **Neural Network**: 90-95% accuracy with high AUC-ROC scores
- **Hyperparameter-tuned XGBoost**: 88-93% accuracy
- **Ensemble Methods**: 85-92% accuracy with robust predictions
- **Basic Models**: 80-88% accuracy

## Advanced Features Explained

### Deep Learning Architecture
- **Input Layer**: 12 features (9 original + 3 engineered)
- **Hidden Layers**: 128 → 64 → 32 → 16 neurons with ReLU activation
- **Output Layer**: Single neuron with sigmoid activation for binary classification
- **Regularization**: Batch Normalization + Dropout to prevent overfitting

### Hyperparameter Tuning
- **Optuna**: Uses Tree-structured Parzen Estimator (TPE) for Bayesian optimization
- **RandomizedSearchCV**: Efficient grid search alternative for large parameter spaces
- **Cross-validation**: 5-fold stratified CV ensures robust evaluation

### Ensemble Learning
- **Voting**: Combines predictions from multiple models (soft voting with probabilities)
- **Stacking**: Learns optimal combination of base models using a meta-learner

