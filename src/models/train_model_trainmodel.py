import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():

    # 1. Define file paths
    input_filepath = "data/processed/multisim_dataset.parquet"
    models_dir = "models"
    output_filepath = os.path.join(models_dir, "model_trainmodel.joblib")

    # 2. Load the processed dataset
    try:
        df = pd.read_parquet(input_filepath)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{input_filepath}'.")
        print("Please run the build_features_trainmodel.py script first.")
        return

    # 3. Split the data into features (X) and target (y)
    target = "target"
    X = df.drop(target, axis=1)
    y = df[target]

    # 4. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
    print("Data successfully split into training and testing sets.")

    # 5. Define the XGBClassifier model with best parameters from the notebook
    best_params = {
        "n_estimators": 334,
        "max_depth": 4,
        "learning_rate": 0.022287980174937348,
        "subsample": 0.6037684234853827,
        "colsample_bytree": 0.6099641378129602,
        "gamma": 1.0375078910112627,
        "lambda": 1.2483253701614754,
        "alpha": 2.270116860679173,
        "random_state": 57,
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**best_params)

    # 6. Train the model ONLY on the training data
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 7. Save the final trained model
    joblib.dump(model, output_filepath)
    print(f"Final model for trainmodel saved to '{output_filepath}'")


if __name__ == "__main__":
    main()
