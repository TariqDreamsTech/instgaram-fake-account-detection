# Instagram Fake Account Detection - ML Application

A comprehensive machine learning application for detecting fake Instagram accounts using advanced ML models, deep learning, and a beautiful web interface. Automatically fetch profile data from Instagram URLs using Apify API.

## 🌟 Features

### 🎨 Modern Web Application
- **Beautiful, Responsive UI** with smooth animations and gradient designs
- **Multiple Input Methods**:
  - Manual entry of account features
  - **Automatic profile fetching from Instagram URLs** using Apify API
- **Real-time Predictions** with multiple model options
- **Interactive Metrics Dashboard** with charts and visualizations
- **Dataset Explorer** with statistics and feature analysis

### 🤖 Machine Learning Models

#### Basic Models
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Model comparison and metrics

#### Advanced Models (2025 Best Practices)
- **Deep Neural Networks** with TensorFlow/Keras
  - Multi-layer architecture (128→64→32→16→1)
  - Batch Normalization and Dropout regularization
  - Early stopping and learning rate scheduling
- **Hyperparameter Tuning**
  - RandomizedSearchCV for Random Forest
  - Optuna Bayesian optimization for XGBoost
  - 5-fold cross-validation
- **Ensemble Methods**
  - Voting Classifier (soft voting)
  - Stacking Classifier with meta-learner
- **Advanced Feature Engineering**
  - Engagement ratio (followers/media)
  - Following-follower ratio
  - Username density features

### 🔗 Instagram Profile URL Integration
- **Automatic Data Fetching** from Instagram profile URLs
- Uses **Apify API** for reliable data extraction
- Supports multiple prediction modes:
  - Best model (Basic or Advanced)
  - All models comparison
  - Single model selection
- Displays fetched profile data and prediction results

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/TariqDreamsTech/instgaram-fake-account-detection.git
cd instgaram-fake-account-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** For advanced features, ensure you have:
- TensorFlow 2.13+ for neural networks
- XGBoost for gradient boosting
- Optuna for hyperparameter optimization

### Step 3: Setup Apify (Optional - for URL fetching)
If you want to use the Instagram profile URL fetching feature:

1. Sign up for a free account at [https://apify.com](https://apify.com)
2. Get your API token from [Settings](https://console.apify.com/account/integrations)
3. Set the environment variable:
   ```bash
   export APIFY_API_TOKEN="your_api_token_here"
   ```
   
   Or create a `.env` file in the project root:
   ```
   APIFY_API_TOKEN=your_api_token_here
   ```

See `ml_app/APIFY_SETUP.md` for detailed setup instructions.

## 🚀 Usage

### Option 1: Web Application (Recommended)

Start the web server:
```bash
cd ml_app
python run_web.py
```

Or using uvicorn directly:
```bash
cd ml_app
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser and navigate to:
```
http://localhost:8000
```

#### Web App Features:
1. **Prediction Tab**
   - Manual entry: Enter account features manually
   - **Profile URL**: Paste any Instagram profile URL to automatically fetch data and predict
   - Multiple prediction modes (Best model, All models, Single model)
   - Beautiful result visualization

2. **Metrics Tab**
   - View all model performance metrics
   - Compare basic and advanced models
   - Interactive charts and tables

3. **Dataset Tab**
   - Explore dataset statistics
   - Feature distribution visualizations
   - Dataset information and metadata

### Option 2: Command-Line Interface

Run the interactive menu:
```bash
cd ml_app
python main.py
```

Menu options:
1. Train Basic Machine Learning Models
2. Train Advanced Models (Deep Learning + Hyperparameter Tuning)
3. Train Both Basic and Advanced Models
4. View Basic Models Metrics
5. View Advanced Models Metrics
6. Predict if an account is fake (Basic Model - Best)
7. Predict if an account is fake (Advanced Model - Best)
8. Predict with All Basic Models
9. Predict with All Advanced Models
10. Exit

### Option 3: Direct Scripts

#### Train Basic Models
```bash
cd ml_app
python train_model.py
```

#### Train Advanced Models
```bash
cd ml_app
python train_advanced.py
```

#### Predict with Basic Model
```bash
cd ml_app
python predict.py
```

#### Predict with Advanced Model
```bash
cd ml_app
python predict_advanced.py
```

## 📊 Model Training

### Basic Models Training

The basic training script:
1. Loads the InstaFake dataset
2. Prepares features (9 features)
3. Splits data into train/test sets (80/20)
4. Trains 4 models:
   - Random Forest
   - Gradient Boosting
   - Logistic Regression
   - SVM
5. Evaluates using accuracy, precision, recall, F1-score
6. Saves all models and metrics to JSON

**Output:**
- Models saved in `models/` directory
- Metrics in `models/all_models_metrics_basic.json`
- Best model: `models/best_model.joblib`

### Advanced Models Training

The advanced training includes:

1. **Hyperparameter Optimization**
   - Random Forest: RandomizedSearchCV (50 iterations)
   - XGBoost: Optuna (30 trials, Bayesian optimization)
   - 5-fold cross-validation

2. **Deep Neural Network**
   - Architecture: 128 → 64 → 32 → 16 → 1 neurons
   - Batch Normalization + Dropout (30%)
   - Adam optimizer with learning rate scheduling
   - Early stopping (patience: 15 epochs)

3. **Ensemble Methods**
   - Voting Classifier (Random Forest + Gradient Boosting + Logistic Regression)
   - Stacking Classifier (Random Forest + Gradient Boosting + SVM → Logistic Regression)

4. **Advanced Features** (12 total)
   - 9 original features
   - 3 engineered features (engagement ratio, following-follower ratio, username density)

5. **Visualizations**
   - Training history plots (`plots/neural_network_training.png`)
   - Feature importance charts (`plots/feature_importance.png`)

**Output:**
- All models saved in `models/` directory
- Metrics in `models/all_models_metrics_advanced.json`
- Neural network: `models/model_advanced_neural_network.h5`

## 🔍 Prediction

### Manual Prediction

Provide these account features:

- `user_media_count`: Number of posts
- `user_follower_count`: Number of followers
- `user_following_count`: Number of accounts following
- `user_has_profil_pic`: Has profile picture (1=Yes, 0=No)
- `user_is_private`: Is private account (1=Yes, 0=No)
- `user_biography_length`: Number of characters in biography
- `username_length`: Number of characters in username
- `username_digit_count`: Number of digits in username

### URL-Based Prediction

Simply paste an Instagram profile URL:
```
https://www.instagram.com/username/
```

The app will:
1. Extract the username from the URL
2. Fetch profile data using Apify API
3. Automatically extract required features
4. Make predictions using selected model(s)
5. Display results with profile information

### Example Usage in Code

```python
from ml_app.predict import predict_from_dict
from ml_app.instagram_fetcher import fetch_profile_from_url

# Method 1: Manual prediction
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

# Method 2: URL-based prediction
profile_result = fetch_profile_from_url("https://www.instagram.com/username/")
if "error" not in profile_result:
    account_data = profile_result["data"]
    result = predict_from_dict(account_data)
    print(f"Prediction: {'FAKE' if result['is_fake'] else 'REAL'}")
```

## 📁 Project Structure

```
fydp-insta/
├── README.md                    # This file
├── requirements.txt            # Root dependencies
├── .gitignore                  # Git ignore rules
│
├── instafake-dataset/          # Dataset directory
│   ├── data/
│   │   └── fake-v1.0/
│   │       ├── fakeAccountData.json
│   │       └── realAccountData.json
│   ├── utils.py                # Dataset utilities
│   └── README.md               # Dataset documentation
│
└── ml_app/                     # Main application directory
    ├── app.py                  # FastAPI web application
    ├── main.py                 # CLI menu application
    ├── run_web.py              # Web server launcher
    │
    ├── train_model.py          # Basic model training
    ├── train_advanced.py        # Advanced training (DL + HP tuning)
    ├── predict.py              # Basic prediction functions
    ├── predict_advanced.py     # Advanced prediction functions
    ├── instagram_fetcher.py    # Apify integration for URL fetching
    │
    ├── requirements.txt        # ML app dependencies
    ├── APIFY_SETUP.md          # Apify setup guide
    ├── WEB_README.md           # Web app documentation
    │
    ├── models/                 # Trained models (created after training)
    │   ├── best_model.joblib
    │   ├── model_basic_*.joblib
    │   ├── model_advanced_*.joblib
    │   ├── model_advanced_neural_network.h5
    │   ├── scaler_basic.joblib
    │   ├── scaler_advanced.joblib
    │   ├── model_info.joblib
    │   ├── model_info_advanced.joblib
    │   ├── all_models_metrics_basic.json
    │   └── all_models_metrics_advanced.json
    │
    ├── plots/                  # Visualization outputs
    │   ├── neural_network_training.png
    │   └── feature_importance.png
    │
    ├── templates/              # Web templates
    │   └── index.html
    │
    └── static/                 # Static web assets
        ├── css/
        │   └── style.css       # Enhanced animations & styling
        └── js/
            └── main.js         # Interactive JavaScript
```

## 🎯 API Endpoints

The web application provides REST API endpoints:

### Training
- `POST /train/basic` - Train basic models
- `POST /train/advanced` - Train advanced models
- `POST /train/both` - Train both

### Metrics
- `GET /metrics/basic` - Get basic models metrics
- `GET /metrics/advanced` - Get advanced models metrics
- `GET /dataset/info` - Get dataset information

### Predictions (Manual Entry)
- `POST /predict/basic/best` - Predict with best basic model
- `POST /predict/advanced/best` - Predict with best advanced model
- `POST /predict/basic/all` - Predict with all basic models
- `POST /predict/advanced/all` - Predict with all advanced models
- `POST /predict/basic/single` - Predict with specific basic model
- `POST /predict/advanced/single` - Predict with specific advanced model

### Predictions (URL-Based)
- `POST /fetch-profile` - Fetch Instagram profile data
- `POST /predict/from-url/basic` - Fetch and predict (basic)
- `POST /predict/from-url/advanced` - Fetch and predict (advanced)
- `POST /predict/from-url/all` - Fetch and predict (all models)

## 📈 Model Performance

Typical performance metrics:

### Advanced Models
- **Neural Network**: 90-95% accuracy, high AUC-ROC scores
- **XGBoost (Tuned)**: 88-93% accuracy
- **Random Forest (Tuned)**: 85-91% accuracy
- **Ensemble Methods**: 87-92% accuracy with robust predictions

### Basic Models
- **Random Forest**: 83-88% accuracy
- **Gradient Boosting**: 82-87% accuracy
- **Logistic Regression**: 80-85% accuracy
- **SVM**: 78-84% accuracy

## 🔧 Advanced Features Explained

### Deep Learning Architecture
- **Input Layer**: 12 features (9 original + 3 engineered)
- **Hidden Layers**: 128 → 64 → 32 → 16 neurons with ReLU activation
- **Output Layer**: Single neuron with sigmoid for binary classification
- **Regularization**: Batch Normalization + Dropout (30%) to prevent overfitting
- **Optimization**: Adam optimizer with learning rate reduction on plateau

### Hyperparameter Tuning
- **Optuna**: Uses Tree-structured Parzen Estimator (TPE) for Bayesian optimization
- **RandomizedSearchCV**: Efficient alternative to exhaustive grid search
- **Cross-validation**: 5-fold stratified CV ensures robust evaluation

### Ensemble Learning
- **Voting**: Combines predictions from multiple models (soft voting with probabilities)
- **Stacking**: Learns optimal combination using a meta-learner (Logistic Regression)

### Instagram URL Fetching
- **Apify Integration**: Reliable data extraction from Instagram profiles
- **Automatic Feature Extraction**: Maps Instagram data to ML model features
- **Error Handling**: Graceful fallback and informative error messages

## 🎨 UI Features

- **Modern Dark Theme** with gradient accents
- **Smooth Animations** and transitions
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Interactive Charts** using Chart.js
- **Real-time Updates** with loading indicators
- **Beautiful Visualizations** for metrics and predictions

## 📝 Dataset

This project uses the **InstaFake Dataset** from the paper:
["Instagram Fake and Automated Account Detection"](https://arxiv.org/pdf/1910.03090.pdf)

The dataset is located in `instafake-dataset/data/` and includes:
- Fake account data (`fakeAccountData.json`)
- Real account data (`realAccountData.json`)

See `instafake-dataset/README.md` for detailed dataset information.

## 🔐 Environment Variables

Create a `.env` file in the project root (optional):

```env
APIFY_API_TOKEN=your_apify_token_here
```

**Note**: `.env` files are automatically ignored by git (see `.gitignore`).

## 🐛 Troubleshooting

### Apify API Issues
- **"APIFY_API_TOKEN not set"**: Make sure you've set the environment variable
- **Empty results**: Check if the profile is accessible and your Apify account has credits
- See `ml_app/APIFY_SETUP.md` for detailed troubleshooting

### Model Training Issues
- **TensorFlow not found**: Install with `pip install tensorflow`
- **XGBoost not found**: Install with `pip install xgboost`
- **Memory errors**: Reduce batch size or use smaller models

### Web Application Issues
- **Port already in use**: Change port in `run_web.py` or use `--port` flag
- **Static files not loading**: Make sure `static/` and `templates/` directories exist

## 📚 Documentation

- **Web App**: See `ml_app/WEB_README.md`
- **Apify Setup**: See `ml_app/APIFY_SETUP.md`
- **Dataset**: See `instafake-dataset/README.md`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project uses the InstaFake Dataset which is licensed under Attribution-NonCommercial (CC BY-NC 4.0).

## 🙏 Acknowledgments

- **InstaFake Dataset**: Created by Fatih Cagatay Akyon and Esat Kalfaoglu
- **Apify**: For Instagram profile scraping API
- **FastAPI**: Modern web framework
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Machine learning library

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ using Machine Learning and Deep Learning**
