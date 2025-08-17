import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def main():

    # 1. Load the processed dataset created by the build_features script
    processed_filepath = "data/processed/data_usage_production.parquet"
    try:
        df = pd.read_parquet(processed_filepath)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{processed_filepath}'.")
        print("Please run the corresponding build_features script first.")
        return

    # 2. Separate features and target
    target = "data_compl_usg_local_m1"
    X = df.drop(columns=[target])
    y = df[target]

    # 3. Split the data to get a consistent test set for evaluation
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=57)

    # 4. Load the pre-trained model
    model_filepath = "models/model_fe1.joblib"
    try:
        rf_model: RandomForestRegressor = joblib.load(model_filepath)
        print(f"Model loaded successfully from '{model_filepath}'")
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{model_filepath}'.")
        print("Please run the corresponding train_model script first.")
        return

    # 5. Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # 6. Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 7. Print the results
    print("\nModel Evaluation Results")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R² Score: {r2:.3f}")


if __name__ == "__main__":
    main()
