"""
Main application for Fake Account Detection
"""

import os
import sys


def menu():
    """Display main menu"""
    print("\n" + "=" * 60)
    print("Fake Account Detection - Machine Learning App")
    print("=" * 60)
    print("1. Train Basic Machine Learning Models")
    print("2. Train Advanced Models (Deep Learning + Hyperparameter Tuning)")
    print("3. Train Both Basic and Advanced Models")
    print("4. View Basic Models Metrics (JSON)")
    print("5. View Advanced Models Metrics (JSON)")
    print("6. Predict if an account is fake (Basic Model - Best)")
    print("7. Predict if an account is fake (Advanced Model - Best)")
    print("8. Predict with All Basic Models")
    print("9. Predict with All Advanced Models")
    print("10. Exit")
    print("=" * 60)


def view_basic_metrics():
    """View basic models metrics from JSON file"""
    import json
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "models", "all_models_metrics_basic.json")

    if not os.path.exists(json_path):
        print(f"\n✗ Basic models metrics not found!")
        print(f"Please train basic models first (Option 1)")
        return

    try:
        with open(json_path, "r") as f:
            metrics = json.load(f)

        print("\n" + "=" * 60)
        print("Basic Models Metrics")
        print("=" * 60)

        # Show best model
        print(f"\n🏆 Best Model: {metrics.get('best_model', 'N/A')}")
        print(f"Best F1-Score: {metrics.get('best_f1_score', 0):.4f}")

        # Show all models
        print("\n" + "-" * 60)
        print("All Models:")
        print("-" * 60)

        for model_name, model_info in metrics.items():
            if model_name in ["best_model", "best_f1_score", "scaler_file"]:
                continue

            if isinstance(model_info, dict) and "metrics" in model_info:
                m = model_info["metrics"]
                print(f"\n📊 {model_name}:")
                print(f"   File: {model_info.get('model_file', 'N/A')}")
                print(f"   Type: {model_info.get('model_type', 'N/A')}")
                print(f"   Accuracy:  {m.get('accuracy', 0):.4f}")
                print(f"   Precision: {m.get('precision', 0):.4f}")
                print(f"   Recall:    {m.get('recall', 0):.4f}")
                print(f"   F1-Score:  {m.get('f1_score', 0):.4f}")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n✗ Error reading metrics: {e}")


def view_advanced_metrics():
    """View advanced models metrics from JSON file"""
    import json
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "models", "all_models_metrics_advanced.json")

    if not os.path.exists(json_path):
        print(f"\n✗ Advanced models metrics not found!")
        print(f"Please train advanced models first (Option 2)")
        return

    try:
        with open(json_path, "r") as f:
            metrics = json.load(f)

        print("\n" + "=" * 60)
        print("Advanced Models Metrics")
        print("=" * 60)

        # Show best model
        print(f"\n🏆 Best Model: {metrics.get('best_model', 'N/A')}")
        print(f"Best F1-Score: {metrics.get('best_f1_score', 0):.4f}")

        # Show all models
        print("\n" + "-" * 60)
        print("All Models:")
        print("-" * 60)

        for model_name, model_info in metrics.items():
            if model_name in ["best_model", "best_f1_score", "scaler_file"]:
                continue

            if isinstance(model_info, dict) and "metrics" in model_info:
                m = model_info["metrics"]
                print(f"\n📊 {model_name}:")
                print(f"   File: {model_info.get('model_file', 'N/A')}")
                print(f"   Type: {model_info.get('model_type', 'N/A')}")
                print(f"   Accuracy:  {m.get('accuracy', 0):.4f}")
                print(f"   Precision: {m.get('precision', 0):.4f}")
                print(f"   Recall:    {m.get('recall', 0):.4f}")
                print(f"   F1-Score:  {m.get('f1_score', 0):.4f}")
                if "auc" in m:
                    print(f"   AUC-ROC:   {m.get('auc', 0):.4f}")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n✗ Error reading metrics: {e}")


def get_account_features():
    """Get account features from user input"""
    print("\nEnter account features:")
    print("-" * 60)

    try:
        account_data = {
            "user_media_count": int(input("Media count (number of posts): ")),
            "user_follower_count": int(input("Follower count: ")),
            "user_following_count": int(input("Following count: ")),
            "user_has_profil_pic": int(input("Has profile picture? (1=Yes, 0=No): ")),
            "user_is_private": int(input("Is private account? (1=Yes, 0=No): ")),
            "user_biography_length": int(
                input("Biography length (number of characters): ")
            ),
            "username_length": int(input("Username length (number of characters): ")),
            "username_digit_count": int(input("Number of digits in username: ")),
        }
        return account_data
    except ValueError:
        print("Error: Please enter valid numbers")
        return None


def main():
    """Main application loop"""
    while True:
        menu()
        choice = input("\nEnter your choice (1-10): ").strip()

        if choice == "1":
            print("\nStarting basic model training...")
            try:
                from train_model import main as train_models

                train_models()
                print("\nBasic model training completed successfully!")
            except Exception as e:
                print(f"\nError during training: {e}")

        elif choice == "2":
            print("\nStarting advanced model training...")
            print("This includes deep learning, hyperparameter tuning, and ensembles.")
            try:
                from train_advanced import main as train_advanced_models

                train_advanced_models()
                print("\nAdvanced model training completed successfully!")
            except ImportError as e:
                print(
                    f"\nError: Missing required libraries. Please install: pip install -r requirements.txt"
                )
                print(f"Details: {e}")
            except Exception as e:
                print(f"\nError during training: {e}")

        elif choice == "3":
            print("\n" + "=" * 60)
            print("Training Both Basic and Advanced Models")
            print("=" * 60)

            # Train basic models first
            print("\n[1/2] Starting basic model training...")
            try:
                from train_model import main as train_models

                train_models()
                print("\n✓ Basic model training completed successfully!")
            except Exception as e:
                print(f"\n✗ Error during basic training: {e}")
                print("Continuing with advanced training...")

            # Train advanced models
            print("\n" + "=" * 60)
            print("[2/2] Starting advanced model training...")
            print("This includes deep learning, hyperparameter tuning, and ensembles.")
            try:
                from train_advanced import main as train_advanced_models

                train_advanced_models()
                print("\n✓ Advanced model training completed successfully!")
            except ImportError as e:
                print(
                    f"\n✗ Error: Missing required libraries. Please install: pip install -r requirements.txt"
                )
                print(f"Details: {e}")
            except Exception as e:
                print(f"\n✗ Error during advanced training: {e}")

            print("\n" + "=" * 60)
            print("All Training Complete!")
            print("=" * 60)
            print("\nBoth basic and advanced models have been trained.")
            print(
                "You can now view metrics (Options 4 & 5) or make predictions (Options 6 & 7)"
            )

        elif choice == "4":
            print("\nViewing Basic Models Metrics...")
            view_basic_metrics()

        elif choice == "5":
            print("\nViewing Advanced Models Metrics...")
            view_advanced_metrics()

        elif choice == "6":
            print("\nPredict if account is fake (Basic Model)")
            account_data = get_account_features()

            if account_data:
                try:
                    from predict import predict_from_dict, print_prediction

                    result = predict_from_dict(account_data)
                    print_prediction(result)
                except FileNotFoundError as e:
                    print(f"\nError: {e}")
                    print("Please train the model first (Option 1 or 3)")
                except Exception as e:
                    print(f"\nError during prediction: {e}")

        elif choice == "7":
            print("\nPredict if account is fake (Advanced Model)")
            account_data = get_account_features()

            if account_data:
                try:
                    from predict_advanced import (
                        predict_from_dict_advanced,
                        print_prediction_advanced,
                    )

                    result = predict_from_dict_advanced(account_data)
                    print_prediction_advanced(result)
                except FileNotFoundError as e:
                    print(f"\nError: {e}")
                    print("Please train the advanced model first (Option 2 or 3)")
                except Exception as e:
                    print(f"\nError during prediction: {e}")

        elif choice == "8":
            print("\nPredict with All Basic Models")
            account_data = get_account_features()

            if account_data:
                try:
                    from predict import (
                        predict_from_all_basic_models,
                        print_all_predictions,
                    )

                    results = predict_from_all_basic_models(account_data)
                    print_all_predictions(results)
                except FileNotFoundError as e:
                    print(f"\nError: {e}")
                    print("Please train basic models first (Option 1 or 3)")
                except Exception as e:
                    print(f"\nError during prediction: {e}")

        elif choice == "9":
            print("\nPredict with All Advanced Models")
            account_data = get_account_features()

            if account_data:
                try:
                    from predict_advanced import (
                        predict_from_all_advanced_models,
                        print_all_advanced_predictions,
                    )

                    results = predict_from_all_advanced_models(account_data)
                    print_all_advanced_predictions(results)
                except FileNotFoundError as e:
                    print(f"\nError: {e}")
                    print("Please train advanced models first (Option 2 or 3)")
                except Exception as e:
                    print(f"\nError during prediction: {e}")

        elif choice == "10":
            print("\nThank you for using Fake Account Detection App!")
            break

        else:
            print("\nInvalid choice. Please enter 1-10.")


if __name__ == "__main__":
    main()
