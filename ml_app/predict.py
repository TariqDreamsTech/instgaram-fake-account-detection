"""
Predict if an Instagram account is fake or real
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "instafake-dataset"))


def load_model():
    """Load the trained model and related files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")

    model_path = os.path.join(model_dir, "best_model.joblib")
    scaler_path = os.path.join(
        model_dir, "scaler_basic.joblib"
    )  # Updated to new naming
    info_path = os.path.join(model_dir, "model_info.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found! Please train the model first by running train_model.py"
        )

    model = joblib.load(model_path)

    # Load scaler if it exists
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if scaler is None:  # Handle case where None was saved
            scaler = None

    info = joblib.load(info_path)

    return model, scaler, info


def prepare_features(account_data, feature_columns):
    """Prepare features from account data"""
    # Calculate follower_following_ratio
    follower_count = account_data.get("user_follower_count", 0)
    following_count = account_data.get("user_following_count", 0)
    follower_following_ratio = follower_count / max(1, following_count)

    # Create feature dictionary
    features = {
        "user_media_count": account_data.get("user_media_count", 0),
        "user_follower_count": account_data.get("user_follower_count", 0),
        "user_following_count": account_data.get("user_following_count", 0),
        "user_has_profil_pic": account_data.get("user_has_profil_pic", 0),
        "user_is_private": account_data.get("user_is_private", 0),
        "follower_following_ratio": follower_following_ratio,
        "user_biography_length": account_data.get("user_biography_length", 0),
        "username_length": account_data.get("username_length", 0),
        "username_digit_count": account_data.get("username_digit_count", 0),
    }

    # Create DataFrame with correct feature order
    feature_df = pd.DataFrame([features])[feature_columns]

    return feature_df


def predict_account(account_data):
    """Predict if an account is fake"""
    model, scaler, info = load_model()
    feature_columns = info["feature_columns"]

    # Prepare features
    features = prepare_features(account_data, feature_columns)

    # Scale if needed
    if scaler is not None:
        try:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
        except Exception as e:
            # If scaling fails, try without scaling
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
    else:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

    return prediction, probabilities, info["model_name"]


def predict_from_dict(account_dict):
    """Predict from a dictionary of account features"""
    prediction, probabilities, model_name = predict_account(account_dict)

    is_fake = bool(prediction)
    fake_probability = probabilities[1] * 100
    real_probability = probabilities[0] * 100

    result = {
        "is_fake": is_fake,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "model_name": model_name,
    }

    return result


def print_prediction(result):
    """Print prediction results in a readable format"""
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Model: {result['model_name']}")
    print(f"\nPrediction: {'FAKE ACCOUNT' if result['is_fake'] else 'REAL ACCOUNT'}")
    print(f"\nProbabilities:")
    print(f"  Real: {result['real_probability']:.2f}%")
    print(f"  Fake: {result['fake_probability']:.2f}%")
    print("=" * 60)


def predict_from_all_basic_models(account_data):
    """Get predictions from all basic models"""
    import json
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    json_path = os.path.join(model_dir, "all_models_metrics_basic.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "Basic models metrics not found! Please train basic models first."
        )

    # Load metrics to get all models
    with open(json_path, "r") as f:
        metrics = json.load(f)

    results = []
    scaler_path = os.path.join(
        model_dir, metrics.get("scaler_file", "scaler_basic.joblib")
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
            if hasattr(model, "predict_proba"):
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
            else:
                prediction = model.predict(features_scaled)[0]
                probabilities = np.array([0.5, 0.5])

            result = {
                "model_name": model_name,
                "is_fake": bool(prediction),
                "fake_probability": float(probabilities[1] * 100),
                "real_probability": float(probabilities[0] * 100),
                "confidence": float(max(probabilities) * 100),
            }
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not get prediction from {model_name}: {e}")
            continue

    return results


def print_all_predictions(results):
    """Print predictions from all models"""
    print("\n" + "=" * 80)
    print("Predictions from All Basic Models")
    print("=" * 80)

    if not results:
        print("No predictions available.")
        return

    # Sort by confidence
    results_sorted = sorted(results, key=lambda x: x["confidence"], reverse=True)

    print(
        f"\n{'Model':<30} {'Prediction':<15} {'Confidence':<12} {'Fake %':<10} {'Real %':<10}"
    )
    print("-" * 80)

    for result in results_sorted:
        prediction_str = "FAKE" if result["is_fake"] else "REAL"
        print(
            f"{result['model_name']:<30} {prediction_str:<15} {result['confidence']:<11.2f}% {result['fake_probability']:<9.2f}% {result['real_probability']:<9.2f}%"
        )

    # Summary
    fake_count = sum(1 for r in results if r["is_fake"])
    real_count = len(results) - fake_count

    print("\n" + "-" * 80)
    print(
        f"Summary: {fake_count} models predict FAKE, {real_count} models predict REAL"
    )
    print(
        f"Average Confidence: {sum(r['confidence'] for r in results) / len(results):.2f}%"
    )
    print("=" * 80)


def example_usage():
    """Example of how to use the prediction function"""
    print("Example Account Prediction")
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

    result = predict_from_dict(example_account)
    print_prediction(result)


if __name__ == "__main__":
    # Check if model exists
    try:
        example_usage()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run train_model.py first to train the model.")
