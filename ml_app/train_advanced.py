"""
Advanced Machine Learning Training with Deep Learning and Hyperparameter Tuning
Includes Neural Networks, Hyperparameter Optimization, and 2025 Best Practices
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
import warnings

warnings.filterwarnings("ignore")

# Try to import TensorFlow/Keras for deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# Try to import XGBoost
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available. Install with: pip install xgboost")

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "instafake-dataset"))
from utils import import_data


def load_dataset():
    """Load the fake account dataset"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "instafake-dataset", "data")
    dataset_version = "fake-v1.0"

    dataset = import_data(dataset_path, dataset_version)
    df = dataset["dataframe"]

    return df


def prepare_data(df, scale=True, scaler_type="standard"):
    """Prepare features and target variable with advanced preprocessing"""
    # Define feature columns (exclude target)
    feature_columns = [
        "user_media_count",
        "user_follower_count",
        "user_following_count",
        "user_has_profil_pic",
        "user_is_private",
        "follower_following_ratio",
        "user_biography_length",
        "username_length",
        "username_digit_count",
    ]

    # Extract features and target
    X = df[feature_columns].copy()
    y = df["is_fake"].copy()

    # Convert boolean to int if needed
    if y.dtype == bool:
        y = y.astype(int)

    # Advanced feature engineering
    X["engagement_ratio"] = X["user_follower_count"] / (
        X["user_media_count"] + 1
    )  # Avoid division by zero
    X["following_follower_ratio"] = X["user_following_count"] / (
        X["user_follower_count"] + 1
    )
    X["username_density"] = X["username_digit_count"] / (X["username_length"] + 1)

    feature_columns = list(X.columns)

    # Scaling
    if scale:
        if scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        return X, y, feature_columns, scaler

    return X, y, feature_columns, None


def create_neural_network(input_dim, dropout_rate=0.3):
    """Create a deep neural network for binary classification"""
    model = models.Sequential(
        [
            layers.Dense(128, activation="relu", input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate / 2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train a deep neural network with early stopping"""
    if not HAS_TENSORFLOW:
        return None, None

    print("\nTraining Deep Neural Network...")

    model = create_neural_network(X_train.shape[1])

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[early_stopping, reduce_lr],
    )

    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    return {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "history": history,
    }, y_pred


def hyperparameter_tuning_random_forest(X_train, y_train):
    """Hyperparameter tuning for Random Forest using RandomizedSearchCV"""
    print("\nHyperparameter Tuning for Random Forest...")

    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=50,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    random_search.fit(X_train, y_train)

    print(f"  Best F1-Score (CV): {random_search.best_score_:.4f}")
    print(f"  Best Parameters: {random_search.best_params_}")

    return random_search.best_estimator_


def hyperparameter_tuning_xgboost(X_train, y_train):
    """Hyperparameter tuning for XGBoost using Optuna"""
    if not HAS_XGBOOST:
        return None

    print("\nHyperparameter Tuning for XGBoost (using Optuna)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": 42,
        }

        model = xgb.XGBClassifier(**params)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1
        )
        return cv_scores.mean()

    study = optuna.create_study(direction="maximize", study_name="XGBoost")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    print(f"  Best F1-Score (CV): {study.best_value:.4f}")
    print(f"  Best Parameters: {study.best_params}")

    best_model = xgb.XGBClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model


def train_advanced_models(X_train, y_train, X_test, y_test, scaler):
    """Train advanced ML models with hyperparameter tuning"""
    models = {}
    results = {}

    # 1. Hyperparameter-tuned Random Forest
    print("\n" + "=" * 60)
    print("Training Advanced Models")
    print("=" * 60)

    rf_tuned = hyperparameter_tuning_random_forest(X_train, y_train)
    models["Random Forest (Tuned)"] = rf_tuned
    y_pred = rf_tuned.predict(X_test)
    results["Random Forest (Tuned)"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "model": rf_tuned,
    }

    # 2. Hyperparameter-tuned XGBoost
    if HAS_XGBOOST:
        xgb_tuned = hyperparameter_tuning_xgboost(X_train, y_train)
        if xgb_tuned:
            models["XGBoost (Tuned)"] = xgb_tuned
            y_pred = xgb_tuned.predict(X_test)
            results["XGBoost (Tuned)"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "model": xgb_tuned,
            }

    # 3. Gradient Boosting with tuned parameters
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    models["Gradient Boosting (Tuned)"] = gb
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    results["Gradient Boosting (Tuned)"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "model": gb,
    }

    # 4. Ensemble: Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ],
        voting="soft",
    )
    models["Voting Ensemble"] = voting_clf
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    results["Voting Ensemble"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "model": voting_clf,
    }

    # 5. Stacking Classifier
    base_models = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
    ]
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    stacking_clf = StackingClassifier(
        estimators=base_models, final_estimator=meta_model, cv=5
    )
    models["Stacking Ensemble"] = stacking_clf
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    results["Stacking Ensemble"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "model": stacking_clf,
    }

    # Print results
    print("\n" + "=" * 60)
    print("Model Performance Summary")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")

    return models, results


def plot_training_history(history, save_path=None):
    """Plot neural network training history"""
    if history is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Plot loss
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training history saved to: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance for tree-based models"""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(
                range(len(importances)),
                [feature_names[i] for i in indices],
                rotation=45,
            )
            plt.ylabel("Importance")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                print(f"Feature importance plot saved to: {save_path}")
            plt.close()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


def plot_confusion_matrix(y_test, y_pred, model_name, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, model_name, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_all_models(
    models_dict, results, neural_network_result, feature_columns, scaler
):
    """Save all models and their metrics in JSON format"""
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    all_metrics = {}
    best_model_name = None
    best_score = -1

    print("\n" + "=" * 60)
    print("Saving All Models and Metrics")
    print("=" * 60)

    # Save all traditional ML models
    for name, model_obj in models_dict.items():
        model_data = results[name]

        # Save model file (clean filename with advanced prefix)
        model_filename = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        model_filename = f"model_advanced_{model_filename}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        joblib.dump(model_obj, model_path)

        # Prepare metrics (convert numpy types to Python types for JSON)
        metrics = {
            "accuracy": float(model_data["accuracy"]),
            "precision": float(model_data["precision"]),
            "recall": float(model_data["recall"]),
            "f1_score": float(model_data["f1_score"]),
        }

        # Check if best model
        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_model_name = name

        all_metrics[name] = {
            "model_file": model_filename,
            "model_type": "traditional_ml",
            "metrics": metrics,
            "feature_columns": feature_columns,
        }

        print(f"Saved: {name} -> {model_filename}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

    # Save neural network if available
    if neural_network_result and "f1_score" in neural_network_result:
        keras_model_path = os.path.join(model_dir, "model_advanced_neural_network.h5")
        neural_network_result["model"].save(keras_model_path)

        neural_metrics = {
            "accuracy": float(neural_network_result["accuracy"]),
            "precision": float(neural_network_result["precision"]),
            "recall": float(neural_network_result["recall"]),
            "f1_score": float(neural_network_result["f1_score"]),
            "auc": float(neural_network_result["auc"]),
        }

        if neural_metrics["f1_score"] > best_score:
            best_score = neural_metrics["f1_score"]
            best_model_name = "Neural Network"

        all_metrics["Neural Network"] = {
            "model_file": "model_advanced_neural_network.h5",
            "model_type": "neural_network",
            "metrics": neural_metrics,
            "feature_columns": feature_columns,
        }

        print(f"Saved: Neural Network -> model_advanced_neural_network.h5")
        print(f"  F1-Score: {neural_metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:  {neural_metrics['auc']:.4f}")

    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler_advanced.joblib")
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        all_metrics["scaler_file"] = "scaler_advanced.joblib"
    else:
        all_metrics["scaler_file"] = None

    # Mark best model
    all_metrics["best_model"] = best_model_name
    all_metrics["best_f1_score"] = float(best_score)

    # Save all metrics to JSON (separate file for advanced models)
    json_path = os.path.join(model_dir, "all_models_metrics_advanced.json")
    try:
        with open(json_path, "w") as f:
            json.dump(all_metrics, f, indent=4, ensure_ascii=False)
        print(f"\n✓ JSON file created successfully: {json_path}")
    except Exception as e:
        print(f"\n✗ Error saving JSON file: {e}")
        raise

    print("\n" + "=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Best F1-Score: {best_score:.4f}")
    print("=" * 60)
    print(f"\nAll advanced models saved to: {model_dir}/")
    print(f"All metrics saved to JSON: {json_path}")

    # Verify JSON file exists
    if os.path.exists(json_path):
        file_size = os.path.getsize(json_path)
        print(f"✓ JSON file verified (size: {file_size} bytes)")

    # Also save best model info separately for compatibility
    info_path = os.path.join(model_dir, "model_info_advanced.joblib")
    if best_model_name == "Neural Network":
        info = {
            "model_name": best_model_name,
            "feature_columns": feature_columns,
            "is_neural_network": True,
            "metrics": neural_network_result,
        }
    else:
        info = {
            "model_name": best_model_name,
            "feature_columns": feature_columns,
            "is_neural_network": False,
            "metrics": results[best_model_name],
        }
    joblib.dump(info, info_path)

    return best_model_name, json_path


def main():
    """Main training function"""
    print("=" * 60)
    print("Advanced Fake Account Detection - Model Training")
    print("Features: Deep Learning, Hyperparameter Tuning, Ensembles")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Fake accounts: {df['is_fake'].sum()}")
    print(f"Real accounts: {(~df['is_fake']).sum()}")

    # Prepare data with advanced preprocessing
    print("\nPreparing data with advanced feature engineering...")
    X, y, feature_columns, scaler = prepare_data(df, scale=True, scaler_type="standard")
    print(f"Features: {feature_columns}")
    print(f"Number of features: {len(feature_columns)}")

    # Split data
    print("\nSplitting data into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Training set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")

    # Train advanced models
    models, results = train_advanced_models(X_train, y_train, X_test, y_test, scaler)

    # Train neural network
    neural_result = None
    if HAS_TENSORFLOW:
        neural_result, _ = train_neural_network(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Plot training history
        if neural_result and "history" in neural_result:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            plots_dir = os.path.join(script_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            plot_training_history(
                neural_result["history"],
                os.path.join(plots_dir, "neural_network_training.png"),
            )

    # Plot feature importance for best tree-based model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Find best tree-based model
    tree_models = [
        name
        for name in results.keys()
        if "Random Forest" in name or "XGBoost" in name or "Gradient Boosting" in name
    ]
    if tree_models:
        best_tree = max(tree_models, key=lambda x: results[x]["f1_score"])
        plot_feature_importance(
            results[best_tree]["model"],
            feature_columns,
            os.path.join(plots_dir, "feature_importance.png"),
        )

    # Save all models and metrics
    best_model_name, json_path = save_all_models(
        models, results, neural_result, feature_columns, scaler
    )

    print("\n" + "=" * 60)
    print("Advanced Training Complete!")
    print("=" * 60)
    print("\nAll plots saved to: plots/")
    print("All models saved to: models/")
    print(f"All metrics saved to JSON: {json_path}")

    return results, neural_result, best_model_name


if __name__ == "__main__":
    results, neural_result, best_model = main()
