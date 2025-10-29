# 🚖 Uber Fare Prediction System: A Full-Stack Machine Learning Solution

## Project Overview

The **Uber Fare Prediction System** is an end-to-end machine learning project designed to accurately estimate the fare amount for an Uber ride based on various spatio-temporal and environmental factors. This project is implemented as a **full-stack web application** using the **Django** framework, allowing users to input ride parameters and receive a real-time fare prediction.

The system integrates a robust regression model trained on ride data, demonstrating a complete pipeline from data cleaning and feature engineering to model deployment in a production-ready environment.

## 📸 Application Screenshots

**Welcome Page**
![Screenshot 2025-03-20 at 10-17-18 Welcome - Uber Fare Predictor](https://github.com/user-attachments/assets/29205401-3870-4824-8386-1ae5b3dbf533)
**Prediction Input Form**
![Screenshot 2025-03-20 at 10-15-00 Predict Fare - Uber Fare Predictor](https://github.com/user-attachments/assets/c7e6deab-344f-43ac-a81e-decace82e396)
**Fare Prediction Result**
![Screenshot 2025-03-20 at 10-18-46 Fare Result - Uber Fare Predictor](https://github.com/user-attachments/assets/da4bd6e2-fe2c-43a9-b1b7-c2b0d499c96a)

---
### Key Features

*   **Regression Model:** Utilizes a highly accurate machine learning model to predict continuous fare amounts.
*   **Advanced Feature Engineering:** Incorporates complex features like Haversine distance, bearing, and distances to key New York City landmarks (JFK, EWR, LGA airports, and the Statue of Liberty).
*   **Django Web Application:** A robust and scalable backend framework for handling user requests and serving the prediction interface.
*   **Real-time Prediction:** The deployed model provides instantaneous fare estimates through a user-friendly web form.
*   **Modular Code Structure:** Clear separation of the ML model artifacts (`model_files/`) and the web application logic (`predictor/`).

---

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have Python (version 3.x) and `pip` installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Menna-Khalid/Uper-Fare-Predection.git
    cd Uper-Fare-Predection
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application (Local Development)

The application runs on the Django development server.

1.  **Run the Django server:**
    ```bash
    python manage.py runserver
    ```

2.  The application will be accessible in your web browser at `http://127.0.0.1:8000/`.

3.  Navigate to the prediction page to input the required features (e.g., coordinates, time, passenger count, car/traffic/weather conditions) and receive a predicted fare.

---

## 📂 Project Structure

The project is structured as a standard Django application, with dedicated directories for the ML model artifacts and the core prediction application.

```
Uper-Fare-Predection/
├── UberFareProject/          # Django project configuration directory
│   ├── settings.py           # Project settings (includes static file and template configuration)
│   └── urls.py               # Main URL routing for the project
├── predictor/                # Django application for handling prediction logic
│   ├── views.py              # Loads the model and handles the prediction request (ML integration)
│   ├── templates/            # HTML templates for the web interface (home, predict, result)
│   ├── static/               # Static files (e.g., images, CSS)
│   └── urls.py               # URL routing for the 'predictor' app
├── model_files/              # Trained model artifacts
│   ├── Uber_model.pkl        # The serialized Machine Learning model (Regression)
│   ├── X_scaler.pkl          # Feature scaler (StandardScaler) for input data
│   └── y_scaler.pkl          # Target variable scaler for inverse-transforming predictions
├── Uber Fare Predection Model🚖.ipynb # Jupyter Notebook for data analysis, feature engineering, and model training
├── requirements.txt          # List of all Python dependencies
└── manage.py                 # Django's command-line utility for administrative tasks
```

---

## 🧠 Machine Learning Pipeline

The machine learning workflow is detailed in the `Uber Fare Predection Model🚖.ipynb` notebook and implemented for deployment in `predictor/views.py`.

### 1. Data Cleaning and Preprocessing

The notebook focuses on rigorous data cleaning for the New York City taxi dataset:
*   **Outlier Removal:** Filters out invalid entries such as zero or negative fare amounts, and rides with zero distance.
*   **Geographic Filtering:** Enforces realistic boundaries for pickup and drop-off coordinates, focusing the model on the NYC area.

### 2. Feature Engineering

The model's high performance is driven by a rich set of engineered features:
*   **Spatio-Temporal Features:** Extraction of `hour`, `day`, `month`, `year`, and `weekday` from the ride timestamp.
*   **Distance Metrics:** Calculation of the **Haversine Distance** between pickup and drop-off points.
*   **Directional Feature:** Calculation of the **Bearing** (direction of travel) between coordinates.
*   **Landmark Proximity:** Calculation of the distance from the ride coordinates to major NYC landmarks:
    *   John F. Kennedy International Airport (JFK)
    *   Newark Liberty International Airport (EWR)
    *   LaGuardia Airport (LGA)
    *   Statue of Liberty (SOL)
*   **Environmental/Contextual Features:** Integration of external factors like `Car Condition`, `Traffic Condition`, and `Weather` (which are mapped to numerical values in `views.py`).

### 3. Model Training and Deployment

*   **Model Selection:** The notebook explores and selects a suitable **Regression Model** (likely a Gradient Boosting Regressor or a similar ensemble method, based on typical performance in this domain) for fare prediction.
*   **Scaling:** Both the feature matrix (`X`) and the target variable (`y`, fare amount) are scaled using `StandardScaler` to ensure optimal model training.
*   **Model Persistence:** The trained model (`Uber_model.pkl`), the feature scaler (`X_scaler.pkl`), and the target scaler (`y_scaler.pkl`) are saved using `joblib` in the `model_files/` directory.

### 4. Deployment Integration (`predictor/views.py`)

The `views.py` file is the bridge between the web application and the ML model:
1.  It loads all three artifacts (`Uber_model.pkl`, `X_scaler.pkl`, `y_scaler.pkl`) on server startup.
2.  It converts user-friendly string inputs (e.g., 'Very Good', 'Congested Traffic') into the numerical format expected by the model using hardcoded mapping dictionaries.
3.  It constructs the feature vector in the exact order used during training.
4.  It applies the `X_scaler` to the input vector.
5.  It calls `model.predict()`.
6.  It applies the `y_scaler`'s `inverse_transform` to the prediction to get the final, human-readable fare amount in dollars.

---

## 🛠 Dependencies

The project relies on a mix of libraries for data science, web development, and deployment.

| Package | Purpose |
| :--- | :--- |
| `Django` | High-level Python Web framework for rapid development of the full-stack application. |
| `scikit-learn` | Core machine learning library for model training and data preprocessing (scaling). |
| `numpy` | Fundamental package for numerical computing, used for handling array data. |
| `joblib` | Used for efficient serialization and deserialization of the large ML model and scalers. |
| `gunicorn` | A production-ready HTTP server for deploying the Django application. |
| `whitenoise` | Simplifies serving static files in a production environment (used in `settings.py`). |

The full list of dependencies can be found in `requirements.txt`.

---

## 🤝 Contribution

This project is maintained by **Menna-Khalid**.

Feel free to fork the repository, submit pull requests, or open issues for bugs and feature suggestions.

---
