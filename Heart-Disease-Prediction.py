import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import warnings

# Suppress warnings for a cleaner professional output
warnings.filterwarnings('ignore', category=UserWarning)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the heart disease dataset from a specified CSV file.

    Args:
        filepath: The path to the CSV file.

    Returns:
        A pandas DataFrame containing the loaded data, or None if file not found.
    """
    try:
        df = pd.read_csv(filepath)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        print("Please download the 'heart_cleveland_upload.csv' file and update the 'dataset_path' variable.")
        return None

def explore_data(df: pd.DataFrame):
    """
    Performs and visualizes initial exploratory data analysis (EDA).

    Args:
        df: The pandas DataFrame to analyze.
    """
    print("\n--- Data Exploration (EDA) ---")
    print("First 5 rows of the data:")
    print(df.head())

    print("\nData Information (columns and types):")
    df.info()

    # Visualization 1: Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='condition', data=df) 
    plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
    plt.show()

    # Visualization 2: Age Distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.show()

def preprocess_data(df: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    """
    Preprocesses the data by splitting, scaling, and preparing it for modeling.

    Args:
        df: The pandas DataFrame.
        test_size: The proportion of the dataset to include in the test split (e.g., 0.3).
        random_state: The seed for reproducible random splitting.

    Returns:
        A tuple containing: 
        (X_train, X_test, y_train, y_test, X_columns)
    """
    print("\n--- Preprocessing Data ---")
    
    # Define Features (X) and Target (y) using 'condition'
    X = df.drop('condition', axis=1)
    y = df['condition']
    X_columns = X.columns  # Save column names for later

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    print(f"Data split complete: {len(X_train)} training samples, {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test, X_columns

def train_and_compare_models(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series) -> tuple:
    """
    Initializes, trains, and evaluates a dictionary of ML models.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.

    Returns:
        A tuple containing:
        (models_dict, results_dict)
    """
    print("\n--- Training and Comparing Models ---")
    
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {} 

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        print(f"{model_name} trained. Accuracy: {accuracy:.4f}")
    
    return models, results

def plot_model_comparison(results: dict):
    """
    Plots a bar chart comparing the accuracy of the trained models.

    Args:
        results: A dictionary with model names as keys and accuracies as values.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values(), color=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy Score')
    plt.ylim(0.7, 1.0) 
    plt.show()

def evaluate_best_model(models: dict, results: dict, X_test: np.ndarray, y_test: pd.Series, X_columns: list):
    """
    Performs a deep-dive evaluation of the best-performing model.
    Plots Classification Report, Confusion Matrix, ROC Curve, and Feature Importance.

    Args:
        models: Dictionary of trained model objects.
        results: Dictionary of model accuracies.
        X_test, y_test: Testing data and labels.
        X_columns: The names of the feature columns.
    """
    # Automatically find the best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\n--- Deep Dive on Best Model ({best_model_name}) ---")

    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test)[:, 1] 

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['No Disease (0)', 'Disease (1)']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No Disease', 'Predicted Disease'],
                yticklabels=['Actual No Disease', 'Actual Disease'])
    plt.title(f'Confusion Matrix for {best_model_name}')
    plt.show()

    # ROC/AUC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {best_model_name}')
    plt.legend(loc="lower right")
    plt.show()

    # Feature Importance Analysis
    print("\n--- Feature Importance Analysis ---")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importances_df = pd.DataFrame(
            importances,
            index=X_columns,
            columns=['Importance']
        ).sort_values(by='Importance', ascending=False)

        print(f"Key factors predicting heart disease (from {best_model_name}):")
        print(feature_importances_df)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances_df.index, y=feature_importances_df['Importance'])
        plt.title(f'Feature Importance (from {best_model_name})')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=90) 
        plt.show()
    else:
        print(f"The best model ({best_model_name}) does not have a 'feature_importances_' attribute.")

def main():
    """
    Main function to run the entire heart disease prediction pipeline.
    """
    # Define project constants
    DATASET_PATH = 'heart_cleveland_upload.csv'
    TEST_SPLIT_SIZE = 0.3
    RANDOM_SEED = 42

    # Load data
    df = load_data(DATASET_PATH)

    if df is not None:
        # Explore data
        explore_data(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, X_columns = preprocess_data(
            df, 
            TEST_SPLIT_SIZE, 
            RANDOM_SEED
        )
        
        # Train models
        models, results = train_and_compare_models(X_train, y_train, X_test, y_test)
        
        # Visualize comparison
        plot_model_comparison(results)
        
        # Evaluate the best model
        evaluate_best_model(models, results, X_test, y_test, X_columns)
        
        print("\n--- Capstone Project Complete! ---")

# Standard Python entry point
if __name__ == "__main__":
    main()