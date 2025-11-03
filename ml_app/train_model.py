"""
Machine Learning Model Training for Fake Account Detection
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

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


def prepare_data(df):
    """Prepare features and target variable"""
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

    return X, y, feature_columns


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models and compare their performance"""

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42),
    }

    results = {}
    trained_models = {}

    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        if name in ["Logistic Regression", "SVM"]:
            # Scale features for linear models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Store scaler with model
            trained_models[name] = {"model": model, "scaler": scaler}
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_models[name] = {"model": model, "scaler": None}

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

    return results, trained_models


def save_all_models_basic(trained_models, results, feature_columns):
    """Save all basic models and their metrics in JSON format"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    all_metrics = {}
    best_model_name = None
    best_score = -1

    print("\n" + "=" * 60)
    print("Saving All Basic Models and Metrics")
    print("=" * 60)

    # Save all models
    for name, model_data in trained_models.items():
        model_obj = model_data["model"]
        scaler = model_data["scaler"]

        # Save model file (clean filename)
        model_filename = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        model_filename = f"model_basic_{model_filename}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        joblib.dump(model_obj, model_path)

        # Prepare metrics (convert numpy types to Python types for JSON)
        metrics = {
            "accuracy": float(results[name]["accuracy"]),
            "precision": float(results[name]["precision"]),
            "recall": float(results[name]["recall"]),
            "f1_score": float(results[name]["f1_score"]),
        }

        # Check if best model
        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_model_name = name

        all_metrics[name] = {
            "model_file": model_filename,
            "model_type": "basic_ml",
            "metrics": metrics,
            "feature_columns": feature_columns,
            "uses_scaler": scaler is not None,
        }

        print(f"Saved: {name} -> {model_filename}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

    # Save scaler if any model uses it
    scaler_path = os.path.join(model_dir, "scaler_basic.joblib")
    has_scaler = any(
        model_data["scaler"] is not None for model_data in trained_models.values()
    )
    if has_scaler:
        # Save scaler from first model that has one
        for model_data in trained_models.values():
            if model_data["scaler"] is not None:
                joblib.dump(model_data["scaler"], scaler_path)
                all_metrics["scaler_file"] = "scaler_basic.joblib"
                break
    else:
        joblib.dump(None, scaler_path)
        all_metrics["scaler_file"] = None

    # Mark best model
    all_metrics["best_model"] = best_model_name
    all_metrics["best_f1_score"] = float(best_score)

    # Save all metrics to JSON
    json_path = os.path.join(model_dir, "all_models_metrics_basic.json")
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
    print(f"\nAll basic models saved to: {model_dir}/")
    print(f"All metrics saved to JSON: {json_path}")

    # Verify JSON file exists
    if os.path.exists(json_path):
        file_size = os.path.getsize(json_path)
        print(f"✓ JSON file verified (size: {file_size} bytes)")

    # Also save best model info separately for compatibility
    info_path = os.path.join(model_dir, "model_info.joblib")
    info = {
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "metrics": results[best_model_name],
    }
    joblib.dump(info, info_path)

    # Save best model file for compatibility
    best_model_path = os.path.join(model_dir, "best_model.joblib")
    best_model_data = trained_models[best_model_name]
    joblib.dump(best_model_data["model"], best_model_path)

    return best_model_name, json_path


def print_detailed_results(trained_models, results, X_test, y_test):
    """Print detailed classification reports for all models"""
    print("\n" + "=" * 60)
    print("Detailed Classification Reports")
    print("=" * 60)

    for name, model_data in trained_models.items():
        print(f"\n{name}:")
        print("-" * 40)

        model = model_data["model"]
        scaler = model_data["scaler"]

        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


def main():
    """Main training function"""
    print("=" * 60)
    print("Fake Account Detection - Model Training")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Fake accounts: {df['is_fake'].sum()}")
    print(f"Real accounts: {(~df['is_fake']).sum()}")

    # Prepare data
    print("\nPreparing data...")
    X, y, feature_columns = prepare_data(df)
    print(f"Features: {feature_columns}")
    print(f"Number of features: {len(feature_columns)}")

    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train models
    results, trained_models = train_models(X_train, y_train, X_test, y_test)

    # Print detailed results
    print_detailed_results(trained_models, results, X_test, y_test)

    # Save all models and metrics
    best_model_name, json_path = save_all_models_basic(
        trained_models, results, feature_columns
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    return results, best_model_name


if __name__ == "__main__":
    results, best_model = main()
