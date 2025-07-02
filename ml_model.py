import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the dataset
df = pd.read_excel('crypto.xlsx', sheet_name='CryptoData')

def train_model(data):
    """
    Train multiple regression models on the provided dataset, select the best model based on MSE for each target variable,
    and save the best model to disk. Returns a dictionary of the best models' filenames for each target variable.

    Parameters:
    - data: DataFrame containing the input features and target variables.

    Returns:
    - best_models: Dictionary with the target as the key and the best model's filename as the value.
    """

    # Define features (independent variables) and target variables (dependent variables)
    features = [
        'Days_Since_Last_High_7_Days',
        '%_Diff_From_High_Last_7_Days',
        'Days_Since_Last_Low_7_Days',
        '%_Diff_From_Low_Last_7_Days'
    ]
    target_columns = [
        '%_Diff_From_High_Next_5_Days',
        '%_Diff_From_Low_Next_5_Days'
    ]

    # Select feature columns, dropping any rows with missing values
    X = data[features].dropna()

    # Check if there’s sufficient data after dropping NaNs
    if X.shape[0] == 0:
        print("No data available for training after dropping NaNs.")
        return None

    # Dictionary to store the best models' file paths for each target
    best_models = {}

    # Loop through each target column to train models individually
    for target in target_columns:
        print(f"\nTraining models for target: {target}")
        
        # Align target variable (y) with X after removing NaNs
        y = data[target].loc[X.index]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define different regression models to be evaluated
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Support Vector Regressor": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }

        # Dictionary to track each model's performance
        model_performance = {}

        # Train each model and evaluate its performance on the test set
        for name, model in models.items():
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions on the test set
                predictions = model.predict(X_test)

                # Calculate performance metrics
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)

                # Output model performance
                print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                
                # Store performance metrics and model object in dictionary
                model_performance[name] = {
                    "model": model,
                    "mse": mse,
                    "mae": mae,
                    "r2": r2
                }
            except Exception as e:
                print(f"Error training {name} for {target}: {e}")

        # Select the best model based on minimum MSE
        best_model_name = min(model_performance, key=lambda x: model_performance[x]["mse"])
        best_model = model_performance[best_model_name]["model"]
        best_mse = model_performance[best_model_name]["mse"]

        # Save the best model to a file
        model_filename = f"{best_model_name.lower().replace(' ', '_')}_{target.lower().replace('%', '').replace(' ', '_')}_model.pkl"
        joblib.dump(best_model, model_filename)
        print(f"Best model for {target} saved: {best_model_name} with MSE: {best_mse:.4f}")

        # Store the filename of the best model for this target in the dictionary
        best_models[target] = model_filename

    return best_models

def predict_outcomes(input_data, target, best_models):
    """
    Load the best model for a specified target variable from disk and use it to predict outcomes based on new input data.

    Parameters:
    - input_data: List of feature values for a new prediction.
    - target: The target variable for which we want predictions.
    - best_models: Dictionary containing the saved best model filenames for each target.

    Returns:
    - Prediction result or None if no model found.
    """
    
    # Retrieve the best model's path for the specified target
    model_path = best_models.get(target)
    
    if model_path:
        # Load the model from the saved file
        model = joblib.load(model_path)
        
        # Reshape input data to match model input requirements
        input_data = np.array(input_data).reshape(1, -1)
        
        # Return the prediction result
        return model.predict(input_data)
    else:
        print(f"No model found for target: {target}")
        return None

# Main program execution
if __name__ == "__main__":
    # Train models and retrieve dictionary of best models' filenames
    trained_models = train_model(df)
    
    # Sample new input data for prediction (change values as per your needs)
    new_input = [3, -2.5, 4, 1.5]
    
    # Loop through each target variable to make predictions using the best model
    for target in ['%_Diff_From_High_Next_5_Days', '%_Diff_From_Low_Next_5_Days']:
        prediction = predict_outcomes(new_input, target, trained_models)
        if prediction is not None:
            print(f"Predicted outcome for {target}: {prediction[0]}")