import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():

    # 1. Define file paths
    input_filepath = "data/processed/ramen-ratings.parquet"
    models_dir = "models"
    model_filepath = os.path.join(models_dir, "model_fe2.joblib")

    # 2. Load the processed dataset to get the test split
    try:
        df = pd.read_parquet(input_filepath)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{input_filepath}'.")
        print("Please run the build_features_fe2.py script first.")
        return

    # 3. Feature Selection (must be identical to the training script)
    selected_features = [
        "Brand",
        "Country",
        "is_top_ten",
        "variety_emb_18",
        "variety_emb_19",
        "variety_emb_20",
        "variety_emb_23",
        "variety_emb_32",
        "variety_emb_38",
        "variety_emb_42",
        "variety_emb_45",
        "variety_emb_49",
        "variety_emb_53",
        "variety_emb_67",
        "variety_emb_79",
        "variety_emb_85",
        "variety_emb_89",
        "variety_emb_91",
        "variety_emb_92",
        "variety_emb_95",
        "variety_emb_103",
        "variety_emb_108",
        "variety_emb_113",
        "variety_emb_134",
        "variety_emb_138",
        "variety_emb_141",
        "variety_emb_144",
        "variety_emb_152",
        "variety_emb_155",
        "variety_emb_158",
        "variety_emb_160",
        "variety_emb_165",
        "variety_emb_171",
        "variety_emb_172",
        "variety_emb_186",
        "variety_emb_188",
        "variety_emb_189",
        "variety_emb_194",
        "variety_emb_195",
        "variety_emb_207",
        "variety_emb_212",
        "variety_emb_217",
        "variety_emb_226",
        "variety_emb_229",
        "variety_emb_231",
        "variety_emb_250",
        "variety_emb_263",
        "variety_emb_276",
        "variety_emb_283",
        "variety_emb_284",
        "variety_emb_296",
        "variety_emb_300",
        "variety_emb_301",
        "variety_emb_302",
        "variety_emb_303",
        "variety_emb_319",
        "variety_emb_333",
        "variety_emb_344",
        "variety_emb_363",
    ]

    target = "Stars"
    X = df[selected_features]
    y = df[target]

    # 4. Split data to get the exact same test set as used in training
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=57)

    # 5. Load the pre-trained model pipeline
    try:
        pipeline = joblib.load(model_filepath)
        print(f"Model pipeline loaded successfully from '{model_filepath}'")
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{model_filepath}'.")
        print("Please run the train_model_fe2.py script first.")
        return

    # 6. Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # 7. Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 8. Print the results
    print("\nModel Evaluation Results")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")


if __name__ == "__main__":
    main()
