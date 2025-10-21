# Heart Disease Prediction using Machine Learning

This project is a capstone project that combines data exploration, preprocessing, model comparison, and feature analysis to predict the likelihood of heart disease in patients based on real medical attributes.

This project uses the "Heart Disease UCI" dataset to compare four different machine learning models:
* Random Forest
* K-Nearest Neighbors (KNN)
* Decision Tree
* Gaussian Naive Bayes

The primary goal is not just to build an accurate classifier, but also to identify the most significant medical factors that contribute to heart disease, based on the model's insights.

## 🚀 Project Overview

The project follows a complete machine learning pipeline:

1.  **Data Loading:** Loads the `heart.csv` dataset.
2.  **Data Exploration (EDA):** Analyzes the data using visualizations (histograms, bar charts, scatter plots) to understand the features and their distributions.
3.  **Data Preprocessing:** Uses `StandardScaler` to normalize all features, ensuring that all variables are treated equally by the models.
4.  **Model Training & Comparison:** Trains all four models on the same data and compares their accuracy.
5.  **Model Evaluation (Deep Dive):** Selects the best-performing model (Random Forest) and evaluates it in depth using a Classification Report, Confusion Matrix, and ROC/AUC Curve.
6.  **Insight & Analysis:** Extracts the **Feature Importance** from the best model to identify the top medical predictors of heart disease.

## 📊 Dataset

This project uses the **Heart Disease Cleveland UCI** dataset, a famous and widely-used benchmark dataset from the UCI Machine Learning Repository.

It contains 14 attributes, including:
* `age`: Age in years
* `sex`: (1 = male; 0 = female)
* `cp`: Chest pain type
* `trestbps`: Resting blood pressure
* `chol`: Serum cholesterol in mg/dl
* `thalach`: Maximum heart rate achieved
* `target`: The diagnosis (0 = no disease, 1 = disease)

## 📈 Key Findings

(You will fill this in after you run the code!)

* **Best Model:** The Random Forest classifier was the top-performing model, achieving an accuracy of **(XX.X%)**.
* **Key Predictors of Heart Disease:** The analysis of feature importances revealed that the top 3 most significant factors for predicting heart disease are:
    1.  **(Feature 1 - e.g., 'cp')**
    2.  **(Feature 2 - e.g., 'thalach')**
    3.  **(Feature 3 - e.g., 'ca')**

## 🔧 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git)
    cd YOUR-REPOSITORY-NAME
    ```

2.  **Install dependencies:**
    This project requires `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Get the dataset:**
    * Download the `heart.csv` file from [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci).
    * Place the `heart.csv` file in the same directory as your Python script.

4.  **Run the script:**
    ```bash
    python your_script_name.py
    ```
