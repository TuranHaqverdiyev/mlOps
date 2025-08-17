import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def main():

    # 1. Define file paths
    input_filepath = "data/processed/ramen-ratings.parquet"
    models_dir = "models"
    output_filepath = os.path.join(models_dir, "model_fe2.joblib")

    # 2. Load the processed dataset
    try:
        df = pd.read_parquet(input_filepath)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{input_filepath}'.")
        print("Please run the build_features_fe2.py script first.")
        return

    # 3. Feature Selection (using the list from your notebook's LassoCV output)
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

    # 4. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)
    print("Data successfully split into training and testing sets.")

    # 5. Define the final modeling pipeline
    # This includes scaling the features and the RandomForestRegressor
    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=1000, random_state=57, n_jobs=-1)),
        ]
    )

    # 6. Train the pipeline on the training data
    final_pipeline.fit(X_train, y_train)
    print("Final model pipeline training complete.")

    # 7. Save the final trained pipeline
    joblib.dump(final_pipeline, output_filepath)
    print(f"Final model pipeline for fe2 saved to '{output_filepath}'")


if __name__ == "__main__":
    main()
