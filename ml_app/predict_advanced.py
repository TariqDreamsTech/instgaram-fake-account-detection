"""
Advanced Prediction Script for Fake Account Detection
Supports Neural Networks and Advanced ML Models
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Try to import TensorFlow/Keras for deep learning
try:
    import tensorflow as tf
    from tensorflow import keras

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "instafake-dataset"))


def load_advanced_model():
    """Load the trained advanced model and related files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")

    # Try advanced model first, fallback to regular model
    model_path = os.path.join(model_dir, "best_model_advanced.joblib")
    keras_model_path = os.path.join(model_dir, "model_advanced_neural_network.h5")
    scaler_path = os.path.join(model_dir, "scaler_advanced.joblib")
    info_path = os.path.join(model_dir, "model_info_advanced.joblib")

    # Fallback to regular model
    if not os.path.exists(model_path) and not os.path.exists(keras_model_path):
        model_path = os.path.join(model_dir, "best_model.joblib")
        scaler_path = os.path.join(
            model_dir, "scaler_basic.joblib"
        )  # Updated to new naming
        info_path = os.path.join(model_dir, "model_info.joblib")

    if not os.path.exists(model_path) and not os.path.exists(keras_model_path):
        raise FileNotFoundError(
            "Model not found! Please train the model first by running train_advanced.py"
        )

    # Check if neural network
    is_neural = os.path.exists(keras_model_path) and HAS_TENSORFLOW

    if is_neural:
        model = keras.models.load_model(keras_model_path)
    else:
        model = joblib.load(model_path)

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if scaler is None:  # Handle case where None was saved
            scaler = None

    info = None
    if os.path.exists(info_path):
        info = joblib.load(info_path)

    return model, scaler, info, is_neural


def prepare_features(account_data, feature_columns):
    """Prepare features from account data with advanced feature engineering"""
    # Basic features
    follower_count = account_data.get("user_follower_count", 0)
    following_count = account_data.get("user_following_count", 0)
    media_count = account_data.get("user_media_count", 0)
    username_length = account_data.get("username_length", 0)
    username_digit_count = account_data.get("username_digit_count", 0)

    # Calculate basic ratio
    follower_following_ratio = follower_count / max(1, following_count)

    # Advanced feature engineering
    engagement_ratio = follower_count / max(1, media_count)
    following_follower_ratio = following_count / max(1, follower_count)
    username_density = username_digit_count / max(1, username_length)

    # Create feature dictionary
    features = {
        "user_media_count": media_count,
        "user_follower_count": follower_count,
        "user_following_count": following_count,
        "user_has_profil_pic": account_data.get("user_has_profil_pic", 0),
        "user_is_private": account_data.get("user_is_private", 0),
        "follower_following_ratio": follower_following_ratio,
        "user_biography_length": account_data.get("user_biography_length", 0),
        "username_length": username_length,
        "username_digit_count": username_digit_count,
        "engagement_ratio": engagement_ratio,
        "following_follower_ratio": following_follower_ratio,
        "username_density": username_density,
    }

    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in features:
            features[col] = 0

    # Create DataFrame with correct feature order
    feature_df = pd.DataFrame([features])[feature_columns]

    return feature_df


def predict_account_advanced(account_data):
    """Predict if an account is fake using advanced model"""
    model, scaler, info, is_neural = load_advanced_model()

    if info is None:
        raise ValueError("Model info not found. Please retrain the model.")

    feature_columns = info["feature_columns"]

    # Prepare features
    features = prepare_features(account_data, feature_columns)

    # Scale if needed
    if scaler is not None:
        try:
            features_scaled = scaler.transform(features)
        except Exception as e:
            # If scaling fails, try without scaling
            print(f"Warning: Scaling failed, using unscaled features: {e}")
            features_scaled = features
    else:
        features_scaled = features

    # Predict
    if is_neural:
        # Neural network prediction
        try:
            prediction_proba = model.predict(features_scaled, verbose=0)[0][0]
            prediction = int(prediction_proba > 0.5)
            probabilities = np.array([1 - prediction_proba, prediction_proba])
        except Exception as e:
            raise ValueError(f"Neural network prediction failed: {e}")
    else:
        # Traditional ML model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_scaled)[0]
            prediction = model.predict(features_scaled)[0]
        else:
            prediction = model.predict(features_scaled)[0]
            probabilities = np.array([0.5, 0.5])  # Default if no proba

    return prediction, probabilities, info["model_name"], is_neural


def predict_from_dict_advanced(account_dict):
    """Predict from a dictionary of account features using advanced model"""
    prediction, probabilities, model_name, is_neural = predict_account_advanced(
        account_dict
    )

    is_fake = bool(prediction)
    # Convert numpy types to Python native types for JSON serialization
    fake_probability = float(probabilities[1] * 100)
    real_probability = float(probabilities[0] * 100)

    result = {
        "is_fake": is_fake,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "model_name": str(model_name),
        "is_neural_network": bool(is_neural),
        "confidence": float(max(fake_probability, real_probability)),
    }

    return result


def print_prediction_advanced(result):
    """Print prediction results in a readable format"""
    print("\n" + "=" * 60)
    print("Advanced Prediction Results")
    print("=" * 60)
    print(f"Model: {result['model_name']}")
    if result["is_neural_network"]:
        print(f"Type: Deep Neural Network")
    else:
        print(f"Type: {result['model_name']}")
    print(f"\nPrediction: {'FAKE ACCOUNT' if result['is_fake'] else 'REAL ACCOUNT'}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"\nProbabilities:")
    print(f"  Real: {result['real_probability']:.2f}%")
    print(f"  Fake: {result['fake_probability']:.2f}%")
    print("=" * 60)


def batch_predict(account_list):
    """Predict for multiple accounts at once"""
    results = []
    model, scaler, info, is_neural = load_advanced_model()

    if info is None:
        raise ValueError("Model info not found. Please retrain the model.")

    feature_columns = info["feature_columns"]

    # Prepare all features
    features_list = []
    for account_data in account_list:
        features = prepare_features(account_data, feature_columns)
        features_list.append(features)

    # Combine into single DataFrame
    all_features = pd.concat(features_list, ignore_index=True)

    # Scale if needed
    if scaler is not None:
        all_features_scaled = scaler.transform(all_features)
    else:
        all_features_scaled = all_features

    # Predict
    if is_neural:
        predictions_proba = model.predict(all_features_scaled, verbose=0)
        predictions = (predictions_proba > 0.5).astype(int).flatten()
        probabilities = np.array([[1 - p[0], p[0]] for p in predictions_proba])
    else:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(all_features_scaled)
            predictions = model.predict(all_features_scaled)
        else:
            predictions = model.predict(all_features_scaled)
            probabilities = np.array([[0.5, 0.5]] * len(predictions))

    # Format results
    for i, account_data in enumerate(account_list):
        result = {
            "account": account_data.get("username", f"Account_{i+1}"),
            "is_fake": bool(predictions[i]),
            "fake_probability": probabilities[i][1] * 100,
            "real_probability": probabilities[i][0] * 100,
            "confidence": max(probabilities[i]) * 100,
        }
        results.append(result)

    return results


def predict_from_all_advanced_models(account_data):
    """Get predictions from all advanced models"""
    import json
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    json_path = os.path.join(model_dir, "all_models_metrics_advanced.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "Advanced models metrics not found! Please train advanced models first."
        )

    # Load metrics to get all models
    with open(json_path, "r") as f:
        metrics = json.load(f)

    results = []
    scaler_path = os.path.join(
        model_dir, metrics.get("scaler_file", "scaler_advanced.joblib")
    )
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            if scaler is None:
                scaler = None
        except:
            scaler = None

    # Predict with each model
    for model_name, model_info in metrics.items():
        if model_name in ["best_model", "best_f1_score", "scaler_file"]:
            continue

        if not isinstance(model_info, dict) or "model_file" not in model_info:
            continue

        try:
            model_file = model_info["model_file"]
            model_path = os.path.join(model_dir, model_file)

            if not os.path.exists(model_path):
                continue

            # Check if neural network
            is_neural = model_info.get("model_type") == "neural_network"

            if is_neural and HAS_TENSORFLOW:
                model = keras.models.load_model(model_path)
            else:
                model = joblib.load(model_path)

            feature_columns = model_info["feature_columns"]

            # Prepare features
            features = prepare_features(account_data, feature_columns)

            # Scale if needed
            features_scaled = features
            if scaler is not None:
                try:
                    features_scaled = scaler.transform(features)
                except:
                    features_scaled = features

            # Predict
            if is_neural:
                try:
                    prediction_proba = model.predict(features_scaled, verbose=0)[0][0]
                    prediction = int(prediction_proba > 0.5)
                    probabilities = np.array([1 - prediction_proba, prediction_proba])
                except Exception as e:
                    print(
                        f"Warning: Neural network prediction failed for {model_name}: {e}"
                    )
                    continue
            else:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features_scaled)[0]
                    prediction = model.predict(features_scaled)[0]
                else:
                    prediction = model.predict(features_scaled)[0]
                    probabilities = np.array([0.5, 0.5])

            result = {
                "model_name": str(model_name),
                "is_fake": bool(prediction),
                "fake_probability": float(probabilities[1] * 100),
                "real_probability": float(probabilities[0] * 100),
                "confidence": float(max(probabilities) * 100),
                "is_neural_network": bool(is_neural),
            }
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not get prediction from {model_name}: {e}")
            continue

    return results


def print_all_advanced_predictions(results):
    """Print predictions from all advanced models"""
    print("\n" + "=" * 80)
    print("Predictions from All Advanced Models")
    print("=" * 80)

    if not results:
        print("No predictions available.")
        return

    # Sort by confidence
    results_sorted = sorted(results, key=lambda x: x["confidence"], reverse=True)

    print(
        f"\n{'Model':<35} {'Type':<18} {'Prediction':<15} {'Confidence':<12} {'Fake %':<10} {'Real %':<10}"
    )
    print("-" * 110)

    for result in results_sorted:
        prediction_str = "FAKE" if result["is_fake"] else "REAL"
        model_type = (
            "Neural Network"
            if result.get("is_neural_network", False)
            else "Traditional ML"
        )
        print(
            f"{result['model_name']:<35} {model_type:<18} {prediction_str:<15} {result['confidence']:<11.2f}% {result['fake_probability']:<9.2f}% {result['real_probability']:<9.2f}%"
        )

    # Summary
    fake_count = sum(1 for r in results if r["is_fake"])
    real_count = len(results) - fake_count

    print("\n" + "-" * 110)
    print(
        f"Summary: {fake_count} models predict FAKE, {real_count} models predict REAL"
    )
    print(
        f"Average Confidence: {sum(r['confidence'] for r in results) / len(results):.2f}%"
    )

    # Neural network count
    nn_count = sum(1 for r in results if r.get("is_neural_network", False))
    if nn_count > 0:
        print(f"Neural Networks: {nn_count} model(s)")
    print("=" * 110)


def example_usage():
    """Example of how to use the advanced prediction function"""
    print("Advanced Account Prediction Example")
    print("=" * 60)

    # Example account data
    example_account = {
        "user_media_count": 50,
        "user_follower_count": 150,
        "user_following_count": 2000,
        "user_has_profil_pic": 1,
        "user_is_private": 0,
        "user_biography_length": 50,
        "username_length": 10,
        "username_digit_count": 3,
    }

    print("\nExample Account Features:")
    for key, value in example_account.items():
        print(f"  {key}: {value}")

    try:
        result = predict_from_dict_advanced(example_account)
        print_prediction_advanced(result)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run train_advanced.py first to train the advanced model.")


if __name__ == "__main__":
    example_usage()
