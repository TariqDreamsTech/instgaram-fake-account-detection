"""
FastAPI Web Application for Fake Account Detection
"""

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import sys
import json
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import training and prediction functions
try:
    from train_model import main as train_models_basic
    from train_advanced import main as train_models_advanced
    from predict import (
        predict_from_dict,
        predict_from_all_basic_models,
        print_all_predictions,
    )
    from predict_advanced import (
        predict_from_dict_advanced,
        predict_from_all_advanced_models,
        print_all_advanced_predictions,
    )
    from instagram_fetcher import fetch_profile_from_url, fetch_profile_from_username
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")

    # Set fallback functions if import fails
    def fetch_profile_from_url(url: str):
        return {"error": f"Instagram fetcher not available: {e}"}

    def fetch_profile_from_username(username: str):
        return {"error": f"Instagram fetcher not available: {e}"}


app = FastAPI(title="Fake Account Detection - ML App", version="1.0.0")

# Setup templates and static files
base_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")


# Models for request/response
class AccountFeatures(BaseModel):
    user_media_count: int
    user_follower_count: int
    user_following_count: int
    user_has_profil_pic: int
    user_is_private: int
    user_biography_length: int
    username_length: int
    username_digit_count: int


class ProfileURLRequest(BaseModel):
    url: str
    prediction_mode: Optional[str] = "basic-best"


class PredictionResponse(BaseModel):
    model_name: Optional[str] = None
    is_fake: bool
    fake_probability: float
    real_probability: float
    confidence: Optional[float] = None
    is_neural_network: Optional[bool] = None


# Helper functions
def load_basic_metrics():
    """Load basic models metrics"""
    models_dir = base_dir / "models"
    json_path = models_dir / "all_models_metrics_basic.json"

    if not json_path.exists():
        return None

    with open(json_path, "r") as f:
        return json.load(f)


def load_advanced_metrics():
    """Load advanced models metrics"""
    models_dir = base_dir / "models"
    json_path = models_dir / "all_models_metrics_advanced.json"

    if not json_path.exists():
        return None

    with open(json_path, "r") as f:
        return json.load(f)


def get_dataset_info():
    """Load and analyze dataset information"""
    try:
        import pandas as pd
        import numpy as np

        # Load dataset using the same method as training
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, "..", "instafake-dataset", "data")
        dataset_version = "fake-v1.0"

        # Add to path to import utils
        sys.path.append(
            os.path.join(os.path.dirname(__file__), "..", "instafake-dataset")
        )
        from utils import import_data

        dataset = import_data(dataset_path, dataset_version)
        df = dataset["dataframe"]

        # Calculate statistics
        total_samples = len(df)
        fake_count = int(df["is_fake"].sum())
        real_count = int((~df["is_fake"]).sum())
        fake_percentage = round((fake_count / total_samples) * 100, 2)
        real_percentage = round((real_count / total_samples) * 100, 2)

        # Feature statistics
        feature_stats = {}
        numeric_features = [
            "user_media_count",
            "user_follower_count",
            "user_following_count",
            "follower_following_ratio",
            "user_biography_length",
            "username_length",
            "username_digit_count",
        ]

        for feature in numeric_features:
            feature_stats[feature] = {
                "mean": float(df[feature].mean()),
                "median": float(df[feature].median()),
                "std": float(df[feature].std()),
                "min": float(df[feature].min()),
                "max": float(df[feature].max()),
            }

        # Categorical features
        categorical_stats = {
            "user_has_profil_pic": {
                "has_pic": int(df["user_has_profil_pic"].sum()),
                "no_pic": int((~df["user_has_profil_pic"].astype(bool)).sum()),
            },
            "user_is_private": {
                "private": int(df["user_is_private"].sum()),
                "public": int((~df["user_is_private"].astype(bool)).sum()),
            },
        }

        # Feature list
        feature_list = [
            {"name": "user_media_count", "description": "Total number of posts"},
            {"name": "user_follower_count", "description": "Total number of followers"},
            {
                "name": "user_following_count",
                "description": "Total number of followings",
            },
            {
                "name": "user_has_profil_pic",
                "description": "Whether account has profile picture",
            },
            {"name": "user_is_private", "description": "Whether account is private"},
            {
                "name": "follower_following_ratio",
                "description": "Ratio of followers to following",
            },
            {
                "name": "user_biography_length",
                "description": "Number of characters in biography",
            },
            {
                "name": "username_length",
                "description": "Number of characters in username",
            },
            {
                "name": "username_digit_count",
                "description": "Number of digits in username",
            },
        ]

        return {
            "total_samples": total_samples,
            "fake_count": fake_count,
            "real_count": real_count,
            "fake_percentage": fake_percentage,
            "real_percentage": real_percentage,
            "feature_stats": feature_stats,
            "categorical_stats": categorical_stats,
            "feature_list": feature_list,
            "dataset_version": dataset_version,
            "dataset_type": dataset.get("dataset_type", "fake"),
        }
    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return None


def convert_account_to_dict(account: AccountFeatures) -> Dict:
    """Convert Pydantic model to dict"""
    return account.dict()


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    basic_metrics = load_basic_metrics()
    advanced_metrics = load_advanced_metrics()
    dataset_info = get_dataset_info()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "basic_metrics": basic_metrics,
            "advanced_metrics": advanced_metrics,
            "dataset_info": dataset_info,
        },
    )


@app.post("/train/basic", response_class=JSONResponse)
async def train_basic_models():
    """Train basic models"""
    try:
        # Run training in background
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, train_models_basic)
        return {"status": "success", "message": "Basic models trained successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/advanced", response_class=JSONResponse)
async def train_advanced_models():
    """Train advanced models"""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, train_models_advanced)
        return {
            "status": "success",
            "message": "Advanced models trained successfully!",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/both", response_class=JSONResponse)
async def train_both_models():
    """Train both basic and advanced models"""
    try:
        loop = asyncio.get_event_loop()

        # Train basic first
        try:
            await loop.run_in_executor(None, train_models_basic)
        except Exception as e:
            print(f"Basic training error: {e}")

        # Train advanced
        try:
            await loop.run_in_executor(None, train_models_advanced)
        except Exception as e:
            print(f"Advanced training error: {e}")

        return {
            "status": "success",
            "message": "Both basic and advanced models trained successfully!",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/basic", response_class=JSONResponse)
async def get_basic_metrics():
    """Get basic models metrics"""
    metrics = load_basic_metrics()
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Basic models metrics not found. Please train models first.",
        )
    return metrics


@app.get("/metrics/advanced", response_class=JSONResponse)
async def get_advanced_metrics():
    """Get advanced models metrics"""
    metrics = load_advanced_metrics()
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Advanced models metrics not found. Please train models first.",
        )
    return metrics


@app.get("/dataset/info", response_class=JSONResponse)
async def get_dataset_info_endpoint():
    """Get dataset information and statistics"""
    info = get_dataset_info()
    if not info:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found or could not be loaded.",
        )
    return info


@app.post("/predict/basic/best", response_class=JSONResponse)
async def predict_basic_best(account: AccountFeatures):
    """Predict with best basic model"""
    try:
        account_dict = convert_account_to_dict(account)
        result = predict_from_dict(account_dict)
        return {
            "status": "success",
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/advanced/best", response_class=JSONResponse)
async def predict_advanced_best(account: AccountFeatures):
    """Predict with best advanced model"""
    try:
        account_dict = convert_account_to_dict(account)
        result = predict_from_dict_advanced(account_dict)

        # Ensure all values are JSON serializable (convert numpy types to Python types)
        import numpy as np

        serializable_result = {
            "is_fake": bool(result.get("is_fake", False)),
            "fake_probability": float(result.get("fake_probability", 0.0)),
            "real_probability": float(result.get("real_probability", 0.0)),
            "model_name": str(result.get("model_name", "Unknown")),
            "is_neural_network": bool(result.get("is_neural_network", False)),
            "confidence": float(result.get("confidence", 0.0)),
        }

        return {
            "status": "success",
            "data": serializable_result,
        }
    except Exception as e:
        import traceback

        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in predict_advanced_best: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/basic/all", response_class=JSONResponse)
async def predict_basic_all(account: AccountFeatures):
    """Predict with all basic models"""
    try:
        account_dict = convert_account_to_dict(account)
        results = predict_from_all_basic_models(account_dict)
        return {
            "status": "success",
            "data": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/advanced/all", response_class=JSONResponse)
async def predict_advanced_all(account: AccountFeatures):
    """Predict with all advanced models"""
    try:
        account_dict = convert_account_to_dict(account)
        results = predict_from_all_advanced_models(account_dict)

        # Ensure all values are JSON serializable
        import numpy as np

        serializable_results = []

        for result in results:
            serializable_result = {
                "model_name": str(result.get("model_name", "Unknown")),
                "is_fake": bool(result.get("is_fake", False)),
                "fake_probability": float(result.get("fake_probability", 0.0)),
                "real_probability": float(result.get("real_probability", 0.0)),
                "confidence": float(result.get("confidence", 0.0)),
                "is_neural_network": bool(result.get("is_neural_network", False)),
            }
            serializable_results.append(serializable_result)

        return {
            "status": "success",
            "data": serializable_results,
        }
    except Exception as e:
        import traceback

        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in predict_advanced_all: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/basic/single", response_class=JSONResponse)
async def predict_basic_single(
    request: Request,
    user_media_count: int = Form(...),
    user_follower_count: int = Form(...),
    user_following_count: int = Form(...),
    user_has_profil_pic: int = Form(...),
    user_is_private: int = Form(...),
    user_biography_length: int = Form(...),
    username_length: int = Form(...),
    username_digit_count: int = Form(...),
    model_name: str = Form(...),
):
    """Predict with a specific basic model"""
    try:
        account_dict = {
            "user_media_count": user_media_count,
            "user_follower_count": user_follower_count,
            "user_following_count": user_following_count,
            "user_has_profil_pic": user_has_profil_pic,
            "user_is_private": user_is_private,
            "user_biography_length": user_biography_length,
            "username_length": username_length,
            "username_digit_count": username_digit_count,
        }
        metrics = load_basic_metrics()

        if not metrics or model_name not in metrics:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        model_info = metrics[model_name]
        model_file = model_info["model_file"]
        model_path = base_dir / "models" / model_file

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found")

        import joblib
        import pandas as pd
        from predict import prepare_features

        model = joblib.load(model_path)
        feature_columns = model_info["feature_columns"]

        # Prepare features
        features = prepare_features(account_dict, feature_columns)

        # Predict
        if hasattr(model, "predict_proba"):
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
        else:
            prediction = model.predict(features)[0]
            probabilities = [0.5, 0.5]

        result = {
            "model_name": model_name,
            "is_fake": bool(prediction),
            "fake_probability": float(probabilities[1] * 100),
            "real_probability": float(probabilities[0] * 100),
            "confidence": float(max(probabilities) * 100),
        }

        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/advanced/single", response_class=JSONResponse)
async def predict_advanced_single(
    request: Request,
    user_media_count: int = Form(...),
    user_follower_count: int = Form(...),
    user_following_count: int = Form(...),
    user_has_profil_pic: int = Form(...),
    user_is_private: int = Form(...),
    user_biography_length: int = Form(...),
    username_length: int = Form(...),
    username_digit_count: int = Form(...),
    model_name: str = Form(...),
):
    """Predict with a specific advanced model"""
    try:
        account_dict = {
            "user_media_count": user_media_count,
            "user_follower_count": user_follower_count,
            "user_following_count": user_following_count,
            "user_has_profil_pic": user_has_profil_pic,
            "user_is_private": user_is_private,
            "user_biography_length": user_biography_length,
            "username_length": username_length,
            "username_digit_count": username_digit_count,
        }
        metrics = load_advanced_metrics()

        if not metrics or model_name not in metrics:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        model_info = metrics[model_name]
        model_file = model_info["model_file"]
        model_path = base_dir / "models" / model_file

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found")

        import joblib
        import pandas as pd
        from predict_advanced import prepare_features

        # Check if neural network
        is_neural = model_info.get("model_type") == "neural_network"

        if is_neural:
            try:
                from tensorflow import keras

                model = keras.models.load_model(model_path)
            except:
                raise HTTPException(status_code=500, detail="TensorFlow not available")
        else:
            model = joblib.load(model_path)

        feature_columns = model_info["feature_columns"]

        # Prepare features
        features = prepare_features(account_dict, feature_columns)

        # Scale if needed
        scaler_path = (
            base_dir / "models" / metrics.get("scaler_file", "scaler_advanced.joblib")
        )
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            if scaler is not None:
                features = scaler.transform(features)

        # Predict
        if is_neural:
            prediction_proba = model.predict(features, verbose=0)[0][0]
            prediction = int(prediction_proba > 0.5)
            probabilities = [1 - prediction_proba, prediction_proba]
        else:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                prediction = model.predict(features)[0]
            else:
                prediction = model.predict(features)[0]
                probabilities = [0.5, 0.5]

        result = {
            "model_name": str(model_name),
            "is_fake": bool(prediction),
            "fake_probability": float(probabilities[1] * 100),
            "real_probability": float(probabilities[0] * 100),
            "confidence": float(max(probabilities) * 100),
            "is_neural_network": bool(is_neural),
        }

        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch-profile", response_class=JSONResponse)
async def fetch_instagram_profile(request: ProfileURLRequest):
    """Fetch Instagram profile data from URL"""
    try:
        result = fetch_profile_from_url(request.url)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "data": result["data"],
            "username": result["username"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")


@app.post("/predict/from-url/basic", response_class=JSONResponse)
async def predict_from_url_basic(request: ProfileURLRequest):
    """Fetch profile from URL and predict with basic model"""
    try:
        # Fetch profile data
        profile_result = fetch_profile_from_url(request.url)

        if "error" in profile_result:
            raise HTTPException(status_code=400, detail=profile_result["error"])

        # Get account data
        account_data = profile_result["data"]

        # Predict
        result = predict_from_dict(account_data)

        return {
            "status": "success",
            "username": profile_result["username"],
            "profile_data": account_data,
            "prediction": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/from-url/advanced", response_class=JSONResponse)
async def predict_from_url_advanced(request: ProfileURLRequest):
    """Fetch profile from URL and predict with advanced model"""
    try:
        # Fetch profile data
        profile_result = fetch_profile_from_url(request.url)

        if "error" in profile_result:
            raise HTTPException(status_code=400, detail=profile_result["error"])

        # Get account data
        account_data = profile_result["data"]

        # Predict
        result = predict_from_dict_advanced(account_data)

        # Ensure all values are JSON serializable
        serializable_result = {
            "is_fake": bool(result.get("is_fake", False)),
            "fake_probability": float(result.get("fake_probability", 0.0)),
            "real_probability": float(result.get("real_probability", 0.0)),
            "model_name": str(result.get("model_name", "Unknown")),
            "is_neural_network": bool(result.get("is_neural_network", False)),
            "confidence": float(result.get("confidence", 0.0)),
        }

        return {
            "status": "success",
            "username": profile_result["username"],
            "profile_data": account_data,
            "prediction": serializable_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in predict_from_url_advanced: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/from-url/all", response_class=JSONResponse)
async def predict_from_url_all(request: ProfileURLRequest):
    """Fetch profile from URL and predict with all models"""
    try:
        # Fetch profile data
        profile_result = fetch_profile_from_url(request.url)

        if "error" in profile_result:
            raise HTTPException(status_code=400, detail=profile_result["error"])

        # Get account data
        account_data = profile_result["data"]

        # Predict with all models based on mode
        mode = request.prediction_mode or "basic-all"

        if "basic" in mode:
            results = predict_from_all_basic_models(account_data)
        else:
            results = predict_from_all_advanced_models(account_data)

        # Ensure all values are JSON serializable
        serializable_results = []
        for result in results:
            serializable_result = {
                "model_name": str(result.get("model_name", "Unknown")),
                "is_fake": bool(result.get("is_fake", False)),
                "fake_probability": float(result.get("fake_probability", 0.0)),
                "real_probability": float(result.get("real_probability", 0.0)),
                "confidence": float(result.get("confidence", 0.0)),
                "is_neural_network": bool(result.get("is_neural_network", False)),
            }
            serializable_results.append(serializable_result)

        return {
            "status": "success",
            "username": profile_result["username"],
            "profile_data": account_data,
            "predictions": serializable_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error in predict_from_url_all: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
