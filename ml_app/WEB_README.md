# Fake Account Detection - Web Application

A beautiful, modern FastAPI web application for detecting fake Instagram accounts using machine learning.

## Features

- 🎨 **Modern, Eye-Catching UI** with gradient designs and smooth animations
- 🧠 **Model Training** - Train basic, advanced, or both models
- 📊 **Metrics Viewing** - View all model metrics in beautiful tables
- 🔍 **Predictions** - Multiple prediction modes:
  - Best model prediction (Basic or Advanced)
  - All models prediction (Basic or Advanced)
  - Single model selection (Basic or Advanced)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Web Application

```bash
# Option 1: Using the run script
python run_web.py

# Option 2: Using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser and navigate to:
```
http://localhost:8000
```

## Web Interface Features

### 1. Training Tab
- Train Basic Models (Random Forest, Gradient Boosting, etc.)
- Train Advanced Models (Deep Learning, Hyperparameter Tuning)
- Train Both at once

### 2. Metrics Tab
- View all basic models metrics
- View all advanced models metrics
- See accuracy, precision, recall, F1-score for each model

### 3. Prediction Tab
- Enter account features through a user-friendly form
- Select prediction mode:
  - **Best Model**: Use the best performing model
  - **All Models**: Get predictions from all models for comparison
  - **Select Model**: Choose a specific model to use
- View results with beautiful visualizations

## API Endpoints

All menu functionality is available through REST API endpoints:

- `GET /` - Home page
- `POST /train/basic` - Train basic models
- `POST /train/advanced` - Train advanced models
- `POST /train/both` - Train both
- `GET /metrics/basic` - Get basic models metrics
- `GET /metrics/advanced` - Get advanced models metrics
- `POST /predict/basic/best` - Predict with best basic model
- `POST /predict/advanced/best` - Predict with best advanced model
- `POST /predict/basic/all` - Predict with all basic models
- `POST /predict/advanced/all` - Predict with all advanced models
- `POST /predict/basic/single` - Predict with specific basic model
- `POST /predict/advanced/single` - Predict with specific advanced model

## UI Features

- **Dark Theme** with gradient accents
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Smooth Animations** and transitions
- **Real-time Updates** - Status messages and loading indicators
- **Beautiful Tables** - Formatted metrics and comparison tables
- **Interactive Cards** - Click-to-train functionality
- **Modern Forms** - Easy-to-use input forms with validation

## Technologies Used

- **FastAPI** - Modern Python web framework
- **Jinja2** - Template engine
- **HTML5/CSS3** - Modern web standards
- **JavaScript** - Dynamic interactions
- **Font Awesome** - Icons

Enjoy your beautiful ML web application! 🚀

