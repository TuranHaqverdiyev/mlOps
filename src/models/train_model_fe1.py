import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def main():

    # 1. Define file paths
    input_filepath = "data/processed/data_usage_production.parquet"
    models_dir = "models"
    output_filepath = os.path.join(models_dir, "model_fe1.joblib")

    # 2. Load the processed dataset
    try:
        processed_df = pd.read_parquet(input_filepath)
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        print("Please run the build_features_fe1.py script first.")
        return

    # 3. Split the data into features (X) and target (y)
    target = "data_compl_usg_local_m1"
    X = processed_df.drop(target, axis=1)
    y = processed_df[target]

    # 4. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
    print("Data successfully split into training and testing sets.")

    # 5. Define the RandomForestRegressor model
    model = RandomForestRegressor(random_state=57, n_jobs=-1)

    # 6. Train the model ONLY on the training data
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 7. Save the final trained model
    joblib.dump(model, output_filepath)
    print(f"Final model for fe1 saved to '{output_filepath}'")


if __name__ == "__main__":
    main()
