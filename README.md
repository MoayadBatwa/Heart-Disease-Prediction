# Heart Disease Prediction: A Full ML Pipeline

This repository contains a professional, end-to-end machine learning project for predicting heart disease. The code is written in a modular, functional style to demonstrate best practices in data science workflows.

The project loads the "Heart Disease Cleveland UCI" dataset, performs exploratory data analysis, preprocesses the data (including feature scaling), and then trains and compares four different classification models:
* Random Forest
* K-Nearest Neighbors (KNN)
* Decision Tree
* Gaussian Naive Bayes

Finally, the script automatically selects the best-performing model, runs a deep-dive evaluation (including a Classification Report, Confusion Matrix, and ROC/AUC Curve), and performs a **feature importance analysis** to identify the key medical predictors of heart disease.

## 🚀 Project Pipeline

This project is structured as a single, clean script that follows these key steps:
1.  **Data Loading:** Safely loads the dataset with error handling.
2.  **Exploratory Data Analysis (EDA):** Visualizes the target distribution and key features.
3.  **Data Preprocessing:** Scales all features using `StandardScaler` and splits the data into 70% training and 30% testing sets.
4.  **Model Training & Comparison:** Loops through all four models, trains them, and compares their accuracy.
5.  **Automated Model Selection:** Programmatically identifies the model with the highest accuracy for evaluation.
6.  **In-Depth Evaluation:** Generates a full suite of evaluation plots and metrics for the winning model.
7.  **Feature Importance Analysis:** Extracts and plots the `feature_importances_` to find the most significant predictors.

## 📊 Dataset

This project uses the **Heart Disease Cleveland UCI** dataset, a classic benchmark dataset from the UCI Machine Learning Repository.

The key features include `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `thalach` (max heart rate), and others. The target variable is `condition` (0 = No Disease, 1 = Disease).

## 📈 Key Findings

(This section will be populated when you run the script!)

* **Best Model:** The script automatically determines the best model. In this run, the **Random Forest** was the top performer, achieving an accuracy of **0.8111**.
* **Model Performance:** The model showed strong performance, with an **AUC score of 0.90**. The confusion matrix showed it was highly effective at correctly identifying both healthy and at-risk patients.
* **Key Predictors of Heart Disease:** The feature importance analysis revealed that the top 3 most significant medical factors for predicting heart disease are:
    1.  **thalach**
    2.  **thal**
    3.  **oldpeak**

## 🔧 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MoayadBatwa/Heart-Disease-Prediction.git](https://github.com/MoayadBatwa/Heart-Disease-Prediction.git)
    cd Heart-Disease-Prediction
    ```

2.  **Install dependencies:**
    This project requires `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Get the dataset:**
    * Download the `heart_cleveland_upload.csv` file from [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci).
    * Place the `.csv` file in the same directory as the Python script.

4.  **Run the script:**
    Execute the main Python script from your terminal. The script will run the full pipeline from loading to analysis and will display all charts and print all findings.
    ```bash
    python Heart-Disease-Prediction.py
    ```

## 👤 Author

* **Name:** Moayad Batwa
* **GitHub:** [@MoayadBatwa](https://github.com/MoayadBatwa)
* **LinkedIn:** [linkedin.com/in/moayadbatwa](https://www.linkedin.com/in/moayadbatwa/)
